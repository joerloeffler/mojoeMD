"""Real preparation regressions. Run directly; no equilibration or production."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/preparation"
sys.path.insert(0, str(ROOT))

from mojoemd.api import _select_engine, prepare_system

BACKEND = os.environ.get("MOJOEMD_TEST_BACKEND", "modular")

def _load_engine():
    return _select_engine(BACKEND)


@dataclass(frozen=True)
class Case:
    name: str
    config: dict
    rejection: str | None = None


def base_config(protein_ff: str, glyco=False, membrane=False, ligand_ff=None, ncaa=False) -> dict:
    receptor = ("glycoprotein" if glyco else "protein") + ("_ncaa" if ncaa else "")
    config = {
        "receptor": str(FIXTURES / f"{receptor}.pdb"),
        "protein_ff": protein_ff,
        "prepare": True, "equilibrate": False, "produce": False,
        "glycoprotein": glyco, "membrane": membrane, "ncaa": ncaa,
        "solvent_padding": 1.0, "membrane_padding": 1.0,
        "membrane_center_z": 5.0,
        "metal_aware_protonation": True,
        "metal_coordination_restraints": False,
    }
    if ligand_ff:
        config.update(ligand=str(FIXTURES / "ligand.sdf"), ligand_resname="LIG", lig_ff=ligand_ff)
    if ncaa:
        config.update(ncaa_resnames=["AIB"], ncaa_ffxml=[str(FIXTURES / "AIB.xml")])
    return config


def case_name(config: dict) -> str:
    ff = config["protein_ff"].replace("/", "_").removesuffix(".xml")
    features = ["protein"]
    features.extend(k for k in ("glycoprotein", "membrane", "ncaa") if config.get(k))
    if config.get("ligand"):
        features.append(config["lig_ff"])
    return ff + "__" + "_".join(features)


def build_cases(suite="matrix", ligand_ffs=("smirnoff", "espaloma")) -> list[Case]:
    engine = _load_engine()
    cases = []
    if suite in ("smoke", "matrix", "all"):
        # Every exposed protein force field must prepare a real solvated protein.
        for protein_ff in engine.PROTEIN_FORCE_FIELDS:
            config = base_config(protein_ff)
            cases.append(Case(case_name(config), config))
        # Exercise every combination of the four modules with known-compatible
        # protein models and the fixture's Amber-compatible NCAA parameters.
        for protein_ff in ("amber14/protein.ff14SB.xml", "amber19/protein.ff19SB.xml"):
            profiles = itertools.product((False, True), repeat=4)
            for glyco, membrane, ligand, ncaa in profiles:
                if not any((glyco, membrane, ligand, ncaa)):
                    continue  # Already covered by vanilla above.
                if suite == "smoke" and (glyco, membrane, ligand, ncaa) not in (
                    (True, False, False, False), (False, False, True, False),
                    (False, False, False, True), (False, True, False, False),
                    (True, True, True, True),
                ):
                    continue
                for ligand_ff in ligand_ffs if ligand else (None,):
                    config = base_config(protein_ff, glyco, membrane, ligand_ff, ncaa)
                    cases.append(Case(case_name(config), config))
        for protein_ff in ("charmm36.xml", "charmm36_2024.xml"):
            config = base_config(protein_ff, membrane=True)
            cases.append(Case(case_name(config), config))
        if suite != "smoke":
            # The remaining Amber models support ligand and membrane modules,
            # but this fixture's NCAA XML is specific to the SB protein atom types.
            for protein_ff in engine.PROTEIN_FORCE_FIELDS:
                if protein_ff.startswith("charmm") or protein_ff in (
                    "amber14/protein.ff14SB.xml", "amber19/protein.ff19SB.xml",
                ):
                    continue
                for membrane, ligand in ((False, True), (True, False), (True, True)):
                    for ligand_ff in ligand_ffs if ligand else (None,):
                        config = base_config(protein_ff, membrane=membrane, ligand_ff=ligand_ff)
                        cases.append(Case(case_name(config), config))
        # Unsupported combinations must fail specifically at configuration time.
        for protein_ff in engine.PROTEIN_FORCE_FIELDS:
            if protein_ff not in ("amber14/protein.ff14SB.xml", "amber19/protein.ff19SB.xml"):
                config = base_config(protein_ff, glyco=True)
                cases.append(Case("reject__" + case_name(config), config, "GLYCAM glycosylation module requires"))
        for protein_ff in ("charmm36.xml", "charmm36_2024.xml"):
            for ligand, ncaa in ((True, False), (False, True)):
                config = base_config(protein_ff, ligand_ff="smirnoff" if ligand else None, ncaa=ncaa)
                cases.append(Case("reject__" + case_name(config), config, "current ligand/NCAA"))
    if suite in ("stress", "all"):
        for ligand_ff in ligand_ffs:
            # Local 9UTB contains no modeled glycans. Do not label this a glyco test.
            path = ROOT / "dev/membrane/validate_membrane_ligand.json"
            config = base_config("amber14/protein.ff14SB.xml", membrane=True, ligand_ff=ligand_ff)
            if path.exists():
                config.update(json.loads(path.read_text()))
                for key in ("receptor", "ligand"):
                    config[key] = str((path.parent / config[key]).resolve())
            else:
                config.update(receptor=str(path.parent / "protein.pdb"), ligand=str(path.parent / "ligand.sdf"))
            config.update(prepare=True, equilibrate=False, produce=False, lig_ff=ligand_ff, glycoprotein=False)
            cases.append(Case(f"9utb__membrane_ligand_{ligand_ff}", config))
            config = base_config("amber14/protein.ff14SB.xml", glyco=True, membrane=True, ligand_ff=ligand_ff, ncaa=True)
            directory = ROOT / "dev/refactor_retests/ncaa_glycan"
            config.update(receptor=str(directory / "AiB_test_AIB_minimized.pdb"),
                          ncaa_ffxml=[str(directory / "ffxml/AIB.xml")], membrane_center_z=None)
            cases.append(Case(f"full_glycan__membrane_ligand_ncaa_{ligand_ff}", config))
            config = base_config("amber14/protein.ff14SB.xml", ligand_ff=ligand_ff)
            config.update(receptor=str(ROOT / "dev/inputs/protein.pdb"),
                          metal_coordination_restraints=True)
            cases.append(Case(f"metal_site__ligand_{ligand_ff}", config))
    return cases


def validate_outputs(directory: Path, config: dict) -> dict:
    """Check artifacts, retained chemistry, connectivity and finite initial energy."""
    import numpy as np
    from openmm import Context, CustomBondForce, Platform, VerletIntegrator, XmlSerializer
    from openmm.app import PDBFile
    from openmm.unit import femtoseconds, kilojoules_per_mole, nanometer

    pdb = PDBFile(str(directory / "prepared.pdb"))
    system = XmlSerializer.deserialize((directory / "system.xml").read_text())
    atoms = list(pdb.topology.atoms())
    assert len(atoms) == system.getNumParticles(), "Topology/particle count mismatch"
    xyz = np.asarray(pdb.positions.value_in_unit(nanometer))
    assert np.isfinite(xyz).all(), "Nonfinite coordinates"
    box = np.asarray(pdb.topology.getPeriodicBoxVectors().value_in_unit(nanometer))
    assert np.isfinite(box).all() and np.linalg.det(box) > 0, "Invalid periodic box"
    assert min(np.linalg.norm(v) for v in box) > 2.0, "Box too small for 1 nm PME cutoff"
    system_box = np.asarray([v.value_in_unit(nanometer) for v in system.getDefaultPeriodicBoxVectors()])
    assert np.allclose(box, system_box, atol=1e-3), "PDB/System box mismatch"
    residues = list(pdb.topology.residues())
    counts = Counter(r.name for r in residues)
    assert counts["HOH"] > 0, "Missing solvent"
    bonds = {frozenset((a.index, b.index)) for a, b in pdb.topology.bonds()}
    if config.get("ligand"):
        ligand = [r for r in residues if r.name == config.get("ligand_resname", "LIG")]
        assert len(ligand) == 1, "Ligand lost or duplicated"
        from rdkit import Chem
        molecule = Chem.SDMolSupplier(config["ligand"], removeHs=False)[0]
        assert molecule is not None, "Invalid ligand fixture"
        # Inputs can omit hydrogens; ligand preparation adds them.
        expected = Counter(a.GetSymbol() for a in Chem.AddHs(molecule).GetAtoms())
        actual = Counter(a.element.symbol for a in ligand[0].atoms())
        assert actual == expected, f"Ligand atom composition changed: {actual} != {expected}"
    if config.get("membrane"):
        # PDB's three-character residue field truncates POPC to POP.
        lipid_name = config.get("lipid_type", "POPC")[:3]
        assert counts[lipid_name] > 0, "Missing membrane lipids"
    if config.get("ncaa"):
        for name in config["ncaa_resnames"]:
            matches = [r for r in residues if r.name == name]
            assert matches, f"NCAA {name} was lost/replaced"
            for residue in matches:
                aa = {a.name: a for a in residue.atoms()}
                if name == "AIB":
                    assert len(aa) == 13 and "HA" not in aa, "Incorrect AIB atoms/hydrogens"
                for atom_name in ("N", "C"):
                    assert any(aa[atom_name] in (a, b) and a.residue != b.residue
                               for a, b in pdb.topology.bonds()), f"Missing NCAA peptide bond at {atom_name}"
    if config.get("glycoprotein"):
        data = json.loads((directory / "glycan_prepare/glycans.json").read_text())
        assignments = data["glycam_assignments"]
        assert assignments and counts["NLN"], "Glyco case must contain actual linked glycans and NLN"
        for label, assignment in assignments.items():
            _, chain, resid = label.split(":")
            matches = [r for r in residues if r.chain.id == chain and r.id == resid]
            assert len(matches) == 1 and matches[0].name == assignment["template"], f"Glycan lost: {label}"
        for link in data["links"]:
            pair = []
            for number in (1, 2):
                _, chain, resid = link[f"residue{number}"].split(":")
                name = link[f"atom{number}"]
                matched = [a for a in atoms if a.residue.chain.id == chain and a.residue.id == resid and a.name == name]
                assert len(matched) == 1, f"Missing glycan link atom: {link}"
                pair.append(matched[0].index)
            assert frozenset(pair) in bonds, f"Lost covalent glycan linkage: {link}"
    if config.get("metal_coordination_restraints"):
        source = PDBFile(config["receptor"])
        copper = {a.index for a in atoms if a.element.symbol == "Cu"}
        source_copper = sum(a.element.symbol == "Cu" for a in source.topology.atoms())
        assert source_copper > 0 and len(copper) == source_copper, "Copper-site fixture lost its metals"
        restraints = [f for f in system.getForces() if isinstance(f, CustomBondForce)
                      and "k_metal_coordination" in f.getEnergyFunction()]
        assert restraints and restraints[0].getNumBonds() > 0, "Missing metal restraints"
        assert any({int(f.getBondParameters(i)[0]), int(f.getBondParameters(i)[1])} & copper
                   for f in restraints for i in range(f.getNumBonds())), "Restraints do not involve the copper site"
    log = (directory / "pipeline.log").read_text()
    assert "PREPARATION STAGE COMPLETE" in log, "Preparation did not complete"
    assert "STAGE 2:" not in log and "STAGE 3:" not in log, "Unexpected dynamics stage"
    assert not any(directory.glob("*.dcd")) and not any(directory.glob("*.chk")), "Unexpected dynamics artifacts"
    # A context evaluates energy; it does not integrate or minimize the system.
    integrator = VerletIntegrator(1 * femtoseconds)
    context = Context(system, integrator, Platform.getPlatformByName("CPU"), {"Threads": "2"})
    context.setPositions(pdb.positions)
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(kilojoules_per_mole / nanometer)
    assert np.isfinite(energy) and np.isfinite(forces).all(), "Nonfinite initial energy/forces"
    del context, integrator
    return {"atoms": len(atoms), "residues": dict(counts), "potential_energy_kj_mol": float(energy)}


def worker(case_file: Path) -> int:
    case = json.loads(case_file.read_text())
    directory = case_file.parent
    report = {"name": case["name"], "status": "failed"}
    start = time.monotonic()
    try:
        engine = _load_engine()
        # Guard the API independently of the booleans in the fixture config.
        with patch.object(engine, "run_equilibrate", side_effect=AssertionError("Equilibration forbidden")), \
             patch.object(engine, "run_production", side_effect=AssertionError("Production forbidden")):
            if case["rejection"]:
                try:
                    engine.parse_arguments(["--config", str(directory / "config.json")])
                except SystemExit as error:
                    assert error.code == 2, f"Wrong rejection exit status: {error.code}"
                else:
                    raise AssertionError("Unsupported combination was accepted")
                # Parent also verifies the specific error in the captured log.
                assert not (directory / "output").exists(), "Rejection occurred after preparation started"
                report.update(status="passed", expected_rejection=case["rejection"])
            else:
                result = prepare_system(directory / "config.json", backend=BACKEND)
                assert result.stages == ("prepare",)
                report.update(validate_outputs(result.output_dir, case["config"]))
                report["status"] = "passed"
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            raise
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        traceback.print_exc()
    report["seconds"] = round(time.monotonic() - start, 3)
    (directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] == "passed" else 1


def run_case(case: Case, output: Path, timeout: float) -> dict:
    directory = output / case.name
    directory.mkdir()  # Never overwrite an earlier result.
    config = dict(case.config, output_dir=str(directory / "output"), prepare=True, equilibrate=False, produce=False)
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    case_file = directory / "case.json"
    case_file.write_text(json.dumps(asdict(case), indent=2) + "\n")
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OPENMM_CPU_THREADS"):
        env[key] = "2"
    env["MPLCONFIGDIR"] = str(output / "cache/matplotlib")
    env["XDG_CACHE_HOME"] = str(output / "cache")
    start = time.monotonic()
    with (directory / "console.log").open("w") as log:
        try:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(case_file)],
                                     cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            if not (directory / "result.json").exists():
                return {"name": case.name, "status": "failed", "error": f"Worker exited {process.returncode} without result"}
        except subprocess.TimeoutExpired:
            return {"name": case.name, "status": "failed", "error": f"Preparation timed out after {timeout}s",
                    "seconds": round(time.monotonic() - start, 3)}
    report = json.loads((directory / "result.json").read_text())
    if case.rejection and case.rejection not in (directory / "console.log").read_text():
        report.update(status="failed", error="Rejected for a different reason than expected")
    if process.returncode != 0 and report["status"] == "passed":
        report.update(status="failed", error=f"Worker exited {process.returncode}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "matrix", "stress", "all"), default="matrix")
    parser.add_argument("--ligand-ff", action="append", choices=("smirnoff", "espaloma"))
    parser.add_argument("--match", help="Run case names containing this substring")
    parser.add_argument("--output", type=Path, help="New output directory; must not already exist")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800, help="Per-case timeout in seconds")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.worker)
    if args.jobs < 1 or args.timeout <= 0:
        parser.error("jobs and timeout must be positive")
    cases = build_cases(args.suite, tuple(dict.fromkeys(args.ligand_ff or ("smirnoff", "espaloma"))))
    if args.match:
        cases = [case for case in cases if args.match in case.name]
    if not cases:
        parser.error("No cases matched")
    if args.list:
        for case in cases:
            print(case.name + (" [expected rejection]" if case.rejection else ""))
        return 0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or ROOT / "dev/preparation_tests" / timestamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    engine = _load_engine()
    import openmm
    metadata = {"suite": args.suite, "backend": BACKEND, "python": sys.version, "executable": sys.executable,
                "openmm": openmm.__version__, "started_utc": timestamp,
                "engine_sha256": hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
                "fixtures_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in FIXTURES.iterdir() if p.is_file()},
                "cases": [case.name for case in cases]}
    metadata["engine_sources_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (sorted((ROOT / "mojoemd").rglob("*.py"))
                     if BACKEND == "modular" else [ROOT / "mojoeMD.py"])
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Running {len(cases)} preparation-only cases -> {output}", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_case, case, output, args.timeout): case for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            try:
                report = future.result()
            except Exception as error:
                report = {"name": case.name, "status": "failed", "error": f"{type(error).__name__}: {error}"}
            results.append(report)
            (output / "summary.json").write_text(json.dumps(sorted(results, key=lambda r: r["name"]), indent=2) + "\n")
            print(f"{report['status'].upper()}: {case.name}" + (f" — {report['error']}" if report.get("error") else ""), flush=True)
    failed = sum(result["status"] != "passed" for result in results)
    print(f"{len(results)-failed}/{len(results)} passed; {failed} failed. Report: {output / 'summary.json'}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
