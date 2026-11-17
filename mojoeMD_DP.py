#!/usr/bin/env python3
"""Development adapter for preparing CALVADOS coarse-grained simulations.

This module is deliberately independent of ``mojoeMD.py``.  The ``prepare``
command writes a self-contained CALVADOS job but never builds an OpenMM
Context, minimizes, equilibrates, or runs dynamics.  The generated ``build.py``
is an explicit, separate build-only step that serializes the CALVADOS system.
Production is only started by the generated ``run.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CALVADOS_VERSION = "3"
TIMESTEP_PS = 0.010
DEFAULT_PRODUCTION_NS = 500.0
DEFAULT_EQUILIBRATION_NS = 1.0
DEFAULT_SAVE_INTERVAL_PS = 100.0
MINIMUM_BOX_NM = 15.0
BOX_PADDING_NM = 4.5

_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "HID": "H",
    "HIE": "H", "HIP": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T",
    "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass(frozen=True)
class PDBSummary:
    residues: int
    chains: int
    sequence: str
    extent_nm: tuple[float, float, float]


@dataclass(frozen=True)
class PreparedJob:
    name: str
    input_pdb: str
    output_dir: str
    ionic_strength_molar: float
    temperature_kelvin: float
    ph: float
    box_nm: float
    residues: int
    chains: int
    production_ns: float
    equilibration_ns: float
    timestep_ps: float
    production_steps: int
    trajectory_interval_ps: float
    trajectory_interval_steps: int
    calvados_version: str = CALVADOS_VERSION
    restraint_model: str = "harmonic elastic network"
    prepared_only: bool = True


def _yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def _write_yaml_mapping(path: Path, values: dict[str, object]) -> None:
    lines = []
    for key, value in values.items():
        if isinstance(value, (list, tuple)):
            encoded = ", ".join(_yaml_scalar(item) for item in value)
            lines.append(f"{key}: [{encoded}]")
        else:
            lines.append(f"{key}: {_yaml_scalar(value)}")
    path.write_text("\n".join(lines) + "\n")


def inspect_pdb(path: Path) -> PDBSummary:
    """Extract one CALVADOS bead per unique protein residue using its C-alpha."""
    seen: set[tuple[str, str, str]] = set()
    sequence: list[str] = []
    chains: set[str] = set()
    xyz: list[tuple[float, float, float]] = []

    with path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            residue_name = line[17:20].strip().upper()
            if residue_name not in _AA3_TO_1:
                raise ValueError(
                    f"Unsupported CALVADOS residue {residue_name!r} in {path}. "
                    "Noncanonical residues need explicit coarse-grained parameters."
                )
            chain = line[21:22].strip() or "_"
            residue_id = (chain, line[22:26].strip(), line[26:27].strip())
            if residue_id in seen:
                continue
            seen.add(residue_id)
            chains.add(chain)
            sequence.append(_AA3_TO_1[residue_name])
            xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))

    if not xyz:
        raise ValueError(f"No protein C-alpha atoms found in {path}")
    extent = tuple((max(axis) - min(axis)) / 10.0 for axis in zip(*xyz))
    return PDBSummary(len(sequence), len(chains), "".join(sequence), extent)


def write_calvados_pdb(source: Path, destination: Path) -> None:
    """Write a protein-only PDB accepted by CALVADOS/MDAnalysis.

    Atomistic ACE/NME caps and solvent are deliberately omitted because the
    coarse-grained model supplies its own terminal masses and charges.
    Histidine protonation aliases are normalized to CALVADOS's HIS bead.
    """
    lines: list[str] = []
    for line in source.read_text().splitlines(keepends=True):
        if line.startswith(("ATOM  ", "HETATM")):
            residue_name = line[17:20].strip().upper()
            if residue_name not in _AA3_TO_1:
                continue
            if residue_name in {"HID", "HIE", "HIP"}:
                line = line[:17] + "HIS" + line[20:]
            line = "ATOM  " + line[6:]
            lines.append(line)
        elif line.startswith(("TER", "END")):
            lines.append(line)
    if not lines or not any(line.startswith("ATOM  ") for line in lines):
        raise ValueError(f"No CALVADOS-compatible protein atoms found in {source}")
    if not lines[-1].startswith("END"):
        lines.append("END\n")
    destination.write_text("".join(lines))


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not name:
        raise ValueError("The system name is empty after sanitization")
    return name


def _positive_float(value: object, field: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be a positive finite number")
    return number


def load_job_config(config_path: Path) -> dict[str, object]:
    data = json.loads(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    return data


def prepare_job(config_path: Path, *, overwrite: bool = False) -> PreparedJob:
    """Write a CALVADOS job definition without running any simulation stage."""
    data = load_job_config(config_path)
    base = config_path.parent.resolve()
    pdb_value = data.get("input_pdb", data.get("receptor"))
    if not pdb_value:
        raise ValueError("Configuration needs 'input_pdb' (or legacy 'receptor')")
    pdb_path = Path(str(pdb_value)).expanduser()
    if not pdb_path.is_absolute():
        pdb_path = (base / pdb_path).resolve()
    if not pdb_path.is_file():
        raise FileNotFoundError(pdb_path)

    output_value = data.get("output_dir")
    if not output_value:
        raise ValueError("Configuration needs 'output_dir'")
    output_dir = Path(str(output_value)).expanduser()
    if not output_dir.is_absolute():
        output_dir = (base / output_dir).resolve()

    name = _safe_name(str(data.get("name", pdb_path.stem)))
    ionic = _positive_float(data.get("ionic_strength", data.get("ionic", 0.15)), "ionic_strength")
    temperature = _positive_float(data.get("temperature_kelvin", data.get("temperature", 298.0)), "temperature_kelvin")
    ph = float(data.get("ph", data.get("pH", 7.4)))
    production_ns = _positive_float(data.get("prod_ns", DEFAULT_PRODUCTION_NS), "prod_ns")
    equilibration_ns = _positive_float(
        data.get("equil_ns", DEFAULT_EQUILIBRATION_NS), "equil_ns"
    )
    save_ps = _positive_float(data.get("trajectory_interval_ps", DEFAULT_SAVE_INTERVAL_PS), "trajectory_interval_ps")
    summary = inspect_pdb(pdb_path)
    if summary.chains != 1:
        raise ValueError("The development adapter currently supports one protein chain per job")

    requested_box = data.get("box_nm")
    box_nm = (
        _positive_float(requested_box, "box_nm")
        if requested_box is not None
        else max(MINIMUM_BOX_NM, max(summary.extent_nm) + 2.0 * BOX_PADDING_NM)
    )
    if box_nm <= 8.0:
        raise ValueError("box_nm must exceed twice the 4 nm electrostatics cutoff")

    steps = int(round(production_ns * 1000.0 / TIMESTEP_PS))
    equilibration_steps = int(round(equilibration_ns * 1000.0 / TIMESTEP_PS))
    save_steps = int(round(save_ps / TIMESTEP_PS))
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir} (use --overwrite)")
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = output_dir / "input"
    inputs.mkdir(exist_ok=True)
    staged_pdb = inputs / f"{name}.pdb"
    write_calvados_pdb(pdb_path, staged_pdb)
    (inputs / "sequence.fasta").write_text(f">{name}\n{summary.sequence}\n")
    (inputs / "domains.yaml").write_text(f"{name}:\n  - [1, {summary.residues}]\n")

    config = {
        "sysname": name,
        "box": [round(box_nm, 4)] * 3,
        "temp": temperature,
        "ionic": ionic,
        "pH": ph,
        "topol": "center",
        "fixed_lambda": 0,
        "eps_lj": 0.2,
        "cutoff_lj": 2.0,
        "cutoff_yu": 4.0,
        "steps": steps,
        "equilibration_steps": equilibration_steps,
        "wfreq": save_steps,
        "logfreq": save_steps,
        "platform": str(data.get("platform", "CPU")),
        "threads": int(data.get("threads", 1)),
        "gpu_id": int(data.get("gpu_id", 0)),
        "runtime": 0,
        "restart": "checkpoint",
        "frestart": "restart.chk",
        "verbose": True,
        "friction_coeff": 0.01,
        "random_number_seed": data.get("seed", 20260924),
        "report_potential_energy": True,
        "slab_eq": False,
        "bilayer_eq": False,
        "pressure_coupling": False,
        "box_eq": False,
        "ext_force": False,
        "custom_restraints": False,
    }
    _write_yaml_mapping(output_dir / "config.yaml", config)

    # CALVADOS's packaged residue table is resolved in build.py/run.py, avoiding
    # a vendored GPL data copy and keeping this adapter independent.
    components_text = f"""defaults:
  molecule_type: protein
  nmol: 1
  charge_termini: both
  alpha: 0
  fresidues: CALVADOS_PACKAGE_RESIDUES
  ffasta: '{(inputs / 'sequence.fasta').as_posix()}'
  kb: 8033.0
  ext_restraint: false
  restraint: true
  cutoff_restr: 0.9
  pdb_folder: '{inputs.as_posix()}'
  restraint_type: harmonic
  k_harmonic: 700.0
  fdomains: '{(inputs / 'domains.yaml').as_posix()}'
  k_go: 15.0
  use_com: false
  periodic: false
  colabfold: 0
system:
  {name}: {{}}
"""
    (output_dir / "components.yaml").write_text(components_text)

    runner = f'''#!/usr/bin/env python3
"""Generated staged CALVADOS runner for build, equilibration, or production."""
from argparse import ArgumentParser
from copy import deepcopy
from importlib import resources
import os
from pathlib import Path
import yaml
from calvados.sim import Sim

parser = ArgumentParser()
parser.add_argument("--stage", choices=("build", "equilibrate", "produce"), required=True)
args = parser.parse_args()
path = Path(__file__).resolve().parent
with (path / "config.yaml").open() as handle:
    config = yaml.safe_load(handle)
config["platform"] = os.environ.get("MOJOEMD_DP_PLATFORM", config["platform"])
config["threads"] = int(os.environ.get("MOJOEMD_DP_THREADS", config["threads"]))
with (path / "components.yaml").open() as handle:
    components = yaml.safe_load(handle)
components["defaults"]["fresidues"] = str(resources.files("calvados") / "data" / "residues.csv")
run_path = path
if args.stage == "equilibrate":
    run_path = path / "equilibration"
    run_path.mkdir(exist_ok=True)
    config = deepcopy(config)
    config["sysname"] = config["sysname"] + "_equil"
    config["steps"] = config["equilibration_steps"]
    config["restart"] = None
    config["frestart"] = "restart.chk"
elif args.stage == "produce":
    equilibrated = path / "equilibration" / "checkpoint.cif"
    if not equilibrated.is_file():
        raise SystemExit(
            "Missing equilibration/checkpoint.cif; run equilibrate.py successfully first"
        )
    run_path = path / "production"
    run_path.mkdir(exist_ok=True)
    config = deepcopy(config)
    config["restart"] = "cif"
    config["frestart"] = str(equilibrated)

simulation = Sim(str(run_path), config, components)
simulation.build_system()
if args.stage != "build":
    simulation.simulate()
'''
    (output_dir / "run.py").write_text(runner)
    (output_dir / "build.py").write_text(
        "#!/usr/bin/env python3\n"
        "import runpy, sys\n"
        "sys.argv = ['run.py', '--stage', 'build']\n"
        "runpy.run_path(str(__import__('pathlib').Path(__file__).with_name('run.py')), run_name='__main__')\n"
    )
    (output_dir / "equilibrate.py").write_text(
        "#!/usr/bin/env python3\n"
        "import runpy, sys\n"
        "sys.argv = ['run.py', '--stage', 'equilibrate']\n"
        "runpy.run_path(str(__import__('pathlib').Path(__file__).with_name('run.py')), run_name='__main__')\n"
    )
    (output_dir / "production.py").write_text(
        "#!/usr/bin/env python3\n"
        "import runpy, sys\n"
        "sys.argv = ['run.py', '--stage', 'produce']\n"
        "runpy.run_path(str(__import__('pathlib').Path(__file__).with_name('run.py')), run_name='__main__')\n"
    )

    job = PreparedJob(
        name=name,
        input_pdb=str(pdb_path),
        output_dir=str(output_dir),
        ionic_strength_molar=ionic,
        temperature_kelvin=temperature,
        ph=ph,
        box_nm=round(box_nm, 4),
        residues=summary.residues,
        chains=summary.chains,
        production_ns=production_ns,
        equilibration_ns=equilibration_ns,
        timestep_ps=TIMESTEP_PS,
        production_steps=steps,
        trajectory_interval_ps=save_ps,
        trajectory_interval_steps=save_steps,
    )
    (output_dir / "manifest.json").write_text(json.dumps(asdict(job), indent=2) + "\n")
    return job


def prepare_many(configs: Iterable[Path], *, overwrite: bool = False) -> list[PreparedJob]:
    return [prepare_job(path, overwrite=overwrite) for path in configs]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare isolated CALVADOS jobs for mojoeMD")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="write job files only; never run dynamics")
    prepare.add_argument("--config", type=Path, action="append", required=True)
    prepare.add_argument("--overwrite", action="store_true")
    inspect = sub.add_parser("inspect", help="inspect a protein PDB without writing files")
    inspect.add_argument("pdb", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            print(json.dumps(asdict(inspect_pdb(args.pdb)), indent=2))
            return 0
        jobs = prepare_many(args.config, overwrite=args.overwrite)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"mojoeMD_DP: {exc}", file=sys.stderr)
        return 2
    for job in jobs:
        print(f"Prepared {job.name}: {job.ionic_strength_molar:g} M, {job.residues} beads -> {job.output_dir}")
    print("Preparation complete. No CALVADOS build, equilibration, or production was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
