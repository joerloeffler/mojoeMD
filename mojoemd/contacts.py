#!/usr/bin/env python3
"""
contact_analysis.py

Post-processing contact analysis around getcontacts + optional mdtraj/OpenMM-style
interaction-energy estimates.

Main ideas
----------
1. Run getcontacts with --cores 1, using a topology/structure and optional trajectory.
2. Parse getcontacts frequency output.
3. Optionally exclude water-bridge contacts, especially wb2, from downstream mapping.
4. Map contact frequencies onto a PDB B-factor field.
5. Optionally write a PyMOL PML script showing contacts as dashed lines.
6. Optionally compute simple LIE-like nonbonded interaction energies with mdtraj coordinates:
   - one residue on chain A against selection 2
   - individual getcontacts-derived contact pairs

Notes
-----
- getcontacts installation/CLI conventions can vary. This script supports both
  get_dynamic_contacts.py/get_static_contacts.py-style commands and plain
  get_dynamic_contacts/get_static_contacts names.
- The energy calculation is intentionally lightweight and approximate. It uses
  Coulomb + Lennard-Jones if partial charges/sigma/epsilon can be loaded from an
  OpenMM system XML or prmtop. If no parameters are available, energy analysis
  is skipped with a clear message.
- For publication-quality interaction categories, getcontacts remains the source
  of truth. Energies here are screening/diagnostic values, not rigorous MM/PBSA.

Example
-------
python contact_analysis.py -c contact_analysis.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import mdtraj as md
except ImportError as e:
    raise SystemExit("mdtraj is required. Install e.g.: pip install mdtraj") from e

try:
    from openmm import XmlSerializer, unit
    from openmm.app import AmberPrmtopFile
    from openmm import NonbondedForce
    _HAS_OPENMM = True
except Exception:
    _HAS_OPENMM = False


# -----------------------------------------------------------------------------
# getcontacts defaults
# -----------------------------------------------------------------------------


DEFAULT_GETCONTACTS_DIR = "/Users/joe/Software/getcontacts"
DEFAULT_GETCONTACTS_CORES = 1


def normalize_getcontacts_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return getcontacts config with defaults injected.

    Supports both:
      cfg["getcontacts"]["path"] / cfg["getcontacts"]["cores"]
    and legacy top-level:
      cfg["getcontacts_path"] / cfg["getcontacts_cores"]
    """
    gc_cfg = dict(cfg.get("getcontacts", {}) or {})
    gc_cfg.setdefault("path", cfg.get("getcontacts_path", DEFAULT_GETCONTACTS_DIR))
    gc_cfg.setdefault("cores", cfg.get("getcontacts_cores", DEFAULT_GETCONTACTS_CORES))
    return gc_cfg


# -----------------------------------------------------------------------------
# Logging/config
# -----------------------------------------------------------------------------


def setup_logging(outdir: Path, level: str = "INFO") -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "contact_analysis.log"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    logging.info("Logging to: %s", log_path)


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [p.strip() for p in re.split(r"[,\s]+", x) if p.strip()]
    return [str(v).strip() for v in x if str(v).strip()]


def normalize_getcontacts_itypes(itypes: Any, gc_cfg: Dict[str, Any]) -> List[str]:
    """Return getcontacts interaction types as separate CLI tokens.

    Recent getcontacts CLIs accept hb/sb/pc/ps/ts/vdw/all, while older configs in
    this project may include wb/wb2 for downstream filtering. Drop unsupported
    labels before invoking getcontacts; they can still be filtered when parsing
    an existing TSV that contains them.
    """
    requested = _as_list(itypes)
    supported = {
        str(x).lower()
        for x in _as_list(gc_cfg.get("supported_itypes", ["all", "sb", "pc", "ps", "ts", "vdw", "hb"]))
    }
    out = []
    dropped = []

    for itype in requested:
        key = itype.lower()
        if key in supported:
            out.append(key)
        else:
            dropped.append(itype)

    if dropped:
        logging.warning(
            "Dropping interaction type(s) unsupported by this getcontacts CLI: %s",
            ", ".join(dropped),
        )
    if not out:
        raise RuntimeError("No supported getcontacts interaction types remain after filtering.")
    return out


# -----------------------------------------------------------------------------
# PDB helpers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidueKey:
    chain: str
    resname: str
    resid: str

    @property
    def label(self) -> str:
        return f"{self.chain}:{self.resname}:{self.resid}"


@dataclass
class ContactRecord:
    residue_1: ResidueKey
    residue_2: ResidueKey
    frame_count: Optional[int]
    frequency: float
    interaction_type: str = "unknown"
    raw: Dict[str, str] | None = None


def parse_residue_token(token: str) -> ResidueKey:
    """Parse tokens such as A:ASP:100, A:100:ASP, ASP:100:A, A:ASP:100A."""
    token = str(token).strip()
    parts = token.split(":")
    if len(parts) >= 3:
        p0, p1, p2 = parts[0].strip(), parts[1].strip(), parts[2].strip()
        # Common getcontacts: chain:resname:resid
        if len(p1) <= 4 and re.search(r"\d", p2):
            return ResidueKey(chain=p0 or " ", resname=p1.upper(), resid=p2)
        # chain:resid:resname
        if re.search(r"\d", p1):
            return ResidueKey(chain=p0 or " ", resname=p2.upper(), resid=p1)
    m = re.match(r"^([^:]+):([A-Za-z0-9]+)([-+]?\d+[A-Za-z]?)$", token)
    if m:
        return ResidueKey(chain=m.group(1), resname="UNK", resid=m.group(3))
    return ResidueKey(chain="?", resname="UNK", resid=token)


def _residue_matches_mdtraj(res: md.core.topology.Residue, key: ResidueKey) -> bool:
    chain_id = getattr(res.chain, "chain_id", None)
    chain_candidates = {str(res.chain.index), f"chain{res.chain.index}"}
    if chain_id:
        chain_candidates.add(str(chain_id).strip())
    key_chain = key.chain.strip()
    if key_chain and key_chain not in chain_candidates:
        return False
    if str(res.resSeq) != str(key.resid).strip():
        return False
    if key.resname != "UNK" and res.name.upper() != key.resname.upper():
        # Be permissive: residue numbering + chain are often enough across PDB variants.
        return True
    return True


def residue_key_to_atom_indices(traj: md.Trajectory, key: ResidueKey, heavy_only: bool = True) -> np.ndarray:
    out: List[int] = []
    for res in traj.topology.residues:
        if not _residue_matches_mdtraj(res, key):
            continue
        for a in res.atoms:
            if heavy_only and a.element is not None and a.element.symbol.upper() == "H":
                continue
            out.append(a.index)
    return np.array(out, dtype=int)


def vmd_chain_selection_indices(traj: md.Trajectory, selection: str) -> Optional[np.ndarray]:
    """Handle simple VMD-style chain selections, e.g. 'chain A or chain C'.

    getcontacts uses VMD selection syntax, while MDTraj does not understand
    chain IDs this way. Keep this intentionally narrow: if the expression
    contains anything other than chain IDs joined by 'or', fall back to MDTraj.
    """
    expr = selection.strip()
    expr = expr[1:-1].strip() if expr.startswith("(") and expr.endswith(")") else expr

    if "chain" not in expr.lower():
        return None

    chains = re.findall(r"(?:^|\bor\b)\s*chain\s+([A-Za-z0-9_]+)", expr, flags=re.IGNORECASE)
    if not chains:
        return None

    leftover = re.sub(r"(?:^|\bor\b)\s*chain\s+[A-Za-z0-9_]+", "", expr, flags=re.IGNORECASE).strip()
    leftover = re.sub(r"\s+", " ", leftover)
    if leftover:
        return None

    wanted = {chain.strip() for chain in chains}
    out = [
        atom.index
        for atom in traj.topology.atoms
        if str(getattr(atom.residue.chain, "chain_id", "")).strip() in wanted
    ]
    if not out:
        raise RuntimeError(f"Selection returned 0 atoms: {selection!r}")
    return np.asarray(out, dtype=int)


def selection_indices(traj: md.Trajectory, selection: str) -> np.ndarray:
    vmd_chain_indices = vmd_chain_selection_indices(traj, selection)
    if vmd_chain_indices is not None:
        return vmd_chain_indices

    idx = traj.topology.select(selection)
    if idx is None or len(idx) == 0:
        raise RuntimeError(f"Selection returned 0 atoms: {selection!r}")
    return np.asarray(idx, dtype=int)


# -----------------------------------------------------------------------------
# getcontacts execution/parsing
# -----------------------------------------------------------------------------


def find_executable(candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        try:
            p = subprocess.run([c, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if p.returncode in (0, 1, 2):
                return c
        except FileNotFoundError:
            continue
    return None


def resolve_getcontacts_script(gc_cfg: Dict[str, Any], script_name: str, path_names: Sequence[str]) -> str:
    """
    Resolve getcontacts helper script.

    Priority:
      1. explicit getcontacts.executable
      2. getcontacts.path / script_name
      3. default /Users/joe/Software/getcontacts / script_name
      4. PATH fallback, e.g. get_dynamic_contacts.py or get_dynamic_contacts
    """
    explicit = gc_cfg.get("executable")
    if explicit:
        return str(explicit)

    gc_dir = Path(str(gc_cfg.get("path", DEFAULT_GETCONTACTS_DIR))).expanduser()
    script_path = gc_dir / script_name
    if script_path.exists():
        return str(script_path)

    # Useful if user sets path to the bin dir with extensionless wrappers.
    for name in path_names:
        candidate = gc_dir / name
        if candidate.exists():
            return str(candidate)

    exe = find_executable(path_names)
    if exe is not None:
        return exe

    raise RuntimeError(
        f"Could not find {script_name!r}. Checked getcontacts.path={gc_dir} and PATH names: {', '.join(path_names)}"
    )


def getcontacts_python_command(gc_cfg: Dict[str, Any], resolved_script: str) -> List[str]:
    """Return command prefix for a resolved getcontacts script.

    .py scripts are run with Python by default. Wrapper executables are run directly.
    Override Python with getcontacts.python, e.g. "python3" or a conda env Python.
    """
    py = str(gc_cfg.get("python", sys.executable))
    if resolved_script.endswith(".py"):
        return [py, resolved_script]
    return [resolved_script]


def run_getcontacts(cfg: Dict[str, Any], outdir: Path) -> Optional[Path]:
    gc_cfg = normalize_getcontacts_config(cfg)
    if not bool(gc_cfg.get("run", True)):
        existing = gc_cfg.get("existing_tsv") or cfg.get("contacts_tsv")
        return Path(existing) if existing else None

    structure = cfg.get("structure") or cfg.get("topology_pdb") or gc_cfg.get("structure")
    trajectory = cfg.get("trajectory") or gc_cfg.get("trajectory")
    if not structure:
        raise RuntimeError("Need 'structure'/'topology_pdb' for getcontacts.")

    dynamic = bool(gc_cfg.get("dynamic", bool(trajectory)))
    itypes = normalize_getcontacts_itypes(gc_cfg.get("itypes", cfg.get("itypes", "hb,sb")), gc_cfg)
    sele = gc_cfg.get("sele", cfg.get("selection1", "chain A"))
    sele2 = gc_cfg.get("sele2", cfg.get("selection2", "chain B"))
    out_tsv = outdir / str(gc_cfg.get("output_tsv", "getcontacts.tsv"))

    extra_args = _as_list(gc_cfg.get("extra_args", []))

    cores = int(gc_cfg.get("cores", DEFAULT_GETCONTACTS_CORES))
    logging.info("Using getcontacts.path=%s", gc_cfg.get("path", DEFAULT_GETCONTACTS_DIR))
    logging.info("Using getcontacts.cores=%s", cores)

    if dynamic:
        exe = resolve_getcontacts_script(
            gc_cfg,
            script_name="get_dynamic_contacts.py",
            path_names=["get_dynamic_contacts.py", "get_dynamic_contacts"],
        )
        if not trajectory:
            raise RuntimeError("Dynamic getcontacts requires 'trajectory'.")
        cmd = getcontacts_python_command(gc_cfg, exe) + [
            "--topology", str(structure),
            "--trajectory", str(trajectory),
            "--itypes", *itypes,
            "--sele", str(sele),
            "--sele2", str(sele2),
            "--cores", str(cores),
            "--output", str(out_tsv),
        ]
    else:
        exe = resolve_getcontacts_script(
            gc_cfg,
            script_name="get_static_contacts.py",
            path_names=["get_static_contacts.py", "get_static_contacts"],
        )
        cmd = getcontacts_python_command(gc_cfg, exe) + [
            "--structure", str(structure),
            "--itypes", *itypes,
            "--sele", str(sele),
            "--sele2", str(sele2),
            "--cores", str(cores),
            "--output", str(out_tsv),
        ]

    cmd.extend(extra_args)
    logging.info("Running getcontacts: %s", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, check=True)
    logging.info("getcontacts output: %s", out_tsv)
    return out_tsv


def parse_getcontacts_tsv(path: str | Path) -> List[ContactRecord]:
    path = Path(path)
    records: List[ContactRecord] = []
    header: Optional[List[str]] = None
    total_frames: Optional[int] = None
    dynamic_counts: Dict[Tuple[str, str, str], set[int]] = defaultdict(set)
    dynamic_raw: Dict[Tuple[str, str, str], Dict[str, str]] = {}

    with path.open("r", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                m = re.search(r"total_frames\s*:\s*(\d+)", line)
                if m:
                    total_frames = int(m.group(1))
                if "Columns:" in line:
                    cols = line.split("Columns:", 1)[1].strip()
                    header = [x.strip() for x in re.split(r"[\t, ]+", cols) if x.strip()]
                continue
            parts = line.split("\t") if "\t" in line else re.split(r"\s+", line.strip())
            if len(parts) < 2:
                continue

            row: Dict[str, str] = {}
            if header and len(header) == len(parts):
                row = dict(zip(header, parts))
            else:
                # Common formats:
                # frame interaction_type atom_1 atom_2
                # residue_1 residue_2 frequency
                # interaction_type residue_1 residue_2 frame_count frequency
                if len(parts) >= 4 and parts[0].isdigit() and ":" in parts[2] and ":" in parts[3]:
                    row = {"frame": parts[0], "interaction_type": parts[1], "atom_1": parts[2], "atom_2": parts[3]}
                elif len(parts) >= 5 and not ":" in parts[0]:
                    row = {"interaction_type": parts[0], "residue_1": parts[1], "residue_2": parts[2], "frame_count": parts[3], "contact_frequency": parts[4]}
                elif len(parts) >= 4:
                    row = {"residue_1": parts[0], "residue_2": parts[1], "frame_count": parts[2], "contact_frequency": parts[3]}
                else:
                    row = {"residue_1": parts[0], "residue_2": parts[1], "contact_frequency": parts[-1]}

            r1s = row.get("residue_1") or row.get("res1") or row.get("atom_1") or parts[0]
            r2s = row.get("residue_2") or row.get("res2") or row.get("atom_2") or parts[1]
            fc_s = row.get("frame_count") or row.get("frames")
            itype = row.get("interaction_type") or row.get("itype") or row.get("interaction") or "unknown"

            if "frame" in row and ("atom_1" in row or "atom_2" in row):
                try:
                    frame = int(float(row["frame"]))
                except Exception:
                    continue
                r1 = parse_residue_token(r1s)
                r2 = parse_residue_token(r2s)
                pair = sorted((r1.label, r2.label))
                key = (itype, pair[0], pair[1])
                dynamic_counts[key].add(frame)
                dynamic_raw.setdefault(key, row)
                continue

            freq_s = row.get("contact_frequency") or row.get("frequency") or row.get("freq") or parts[-1]

            try:
                freq = float(freq_s)
            except Exception:
                continue
            try:
                frame_count = int(float(fc_s)) if fc_s is not None else None
            except Exception:
                frame_count = None

            records.append(ContactRecord(parse_residue_token(r1s), parse_residue_token(r2s), frame_count, freq, itype, row))

    if dynamic_counts:
        if total_frames is None:
            max_frame = max(frame for frames in dynamic_counts.values() for frame in frames)
            total_frames = max_frame + 1
            logging.warning(
                "No total_frames header found in %s; inferred total_frames=%d from max frame index.",
                path,
                total_frames,
            )

        for (itype, r1_label, r2_label), frames in dynamic_counts.items():
            frame_count = len(frames)
            freq = frame_count / float(total_frames) if total_frames else 0.0
            records.append(
                ContactRecord(
                    parse_residue_token(r1_label),
                    parse_residue_token(r2_label),
                    frame_count,
                    freq,
                    itype,
                    dynamic_raw.get((itype, r1_label, r2_label)),
                )
            )
        logging.info(
            "Aggregated %d dynamic contact observations into %d residue-pair frequency records using total_frames=%d.",
            sum(len(frames) for frames in dynamic_counts.values()),
            len(dynamic_counts),
            total_frames,
        )

    logging.info("Parsed %d contact records from %s", len(records), path)
    return records


def filter_contacts(records: List[ContactRecord], cfg: Dict[str, Any]) -> List[ContactRecord]:
    excluded = {x.lower() for x in _as_list(cfg.get("exclude_itypes", []))}
    if bool(cfg.get("exclude_wb2", False)):
        excluded.add("wb2")
    min_freq = float(cfg.get("min_frequency", 0.0))
    out = [r for r in records if r.frequency >= min_freq and r.interaction_type.lower() not in excluded]
    logging.info("Filtered contacts: %d -> %d", len(records), len(out))
    return out


def write_contacts_csv(records: List[ContactRecord], out_csv: Path) -> None:
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["interaction_type", "residue_1", "residue_2", "frame_count", "frequency"])
        for r in records:
            w.writerow([r.interaction_type, r.residue_1.label, r.residue_2.label, "" if r.frame_count is None else r.frame_count, f"{r.frequency:.6f}"])
    logging.info("Wrote contacts CSV: %s", out_csv)


# -----------------------------------------------------------------------------
# Mapping contacts onto PDB B-factor
# -----------------------------------------------------------------------------


def residue_contact_scores(records: List[ContactRecord]) -> Dict[Tuple[str, str], float]:
    scores: Dict[Tuple[str, str], float] = defaultdict(float)
    for r in records:
        scores[(r.residue_1.chain, r.residue_1.resid)] += float(r.frequency)
        scores[(r.residue_2.chain, r.residue_2.resid)] += float(r.frequency)
    return scores


def write_contact_mapped_pdb(top_pdb: str | Path, out_pdb: Path, records: List[ContactRecord], scale: float = 100.0) -> None:
    scores = residue_contact_scores(records)
    with open(top_pdb, "r", errors="ignore") as fin, out_pdb.open("w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                chain = line[21].strip() or " "
                resid = line[22:27].strip()  # includes insertion code column if present by width overlap
                resid = resid.replace(" ", "")
                resseq = line[22:26].strip()
                icode = line[26].strip()
                candidates = [(chain, resid), (chain, resseq + icode), (chain, resseq)]
                val = 0.0
                for k in candidates:
                    if k in scores:
                        val = scores[k] * scale
                        break
                newline = line[:60] + f"{val:6.2f}" + line[66:]
                fout.write(newline)
            else:
                fout.write(line)
    logging.info("Wrote contact-mapped PDB: %s", out_pdb)


# -----------------------------------------------------------------------------
# PML visualization
# -----------------------------------------------------------------------------


def write_contacts_pml(
    records: List[ContactRecord],
    out_pml: Path,
    object_name: str = "contact_lines",
    max_contacts: int = 200,
    min_width: float = 1.5,
    max_width: float = 10.0,
) -> None:
    recs = sorted(records, key=lambda r: r.frequency, reverse=True)[:int(max_contacts)]
    max_freq = max((r.frequency for r in recs), default=1.0)
    with out_pml.open("w") as f:
        f.write("# Load your structure first, then run this script in PyMOL.\n")
        f.write("set dash_gap, 0.05\nset dash_radius, 0.08\nset dash_color, black\n")
        f.write("hide labels\n")
        for i, r in enumerate(recs, start=1):
            n1 = f"ct_{i:04d}_a"
            n2 = f"ct_{i:04d}_b"
            dname = f"ct_{i:04d}_{r.interaction_type}"
            f.write(f"select {n1}, chain {r.residue_1.chain} and resi {r.residue_1.resid} and name CA+C4*+C4'\n")
            f.write(f"select {n2}, chain {r.residue_2.chain} and resi {r.residue_2.resid} and name CA+C4*+C4'\n")
            f.write(f"distance {dname}, {n1}, {n2}\n")
            width = min_width + (max_width - min_width) * math.sqrt(r.frequency / max_freq)
            f.write(f"set dash_width, {width:.2f}, {dname}\n")
            f.write(f"color black, {dname}\n")
            f.write(f"delete {n1}\ndelete {n2}\n")
        f.write("hide labels\n")
    logging.info("Wrote PyMOL contact script: %s", out_pml)


# -----------------------------------------------------------------------------
# Energy analysis
# -----------------------------------------------------------------------------


@dataclass
class NBParams:
    charges_e: np.ndarray
    sigma_nm: np.ndarray
    epsilon_kj_mol: np.ndarray


def load_nonbonded_params(cfg: Dict[str, Any], traj: md.Trajectory) -> Optional[NBParams]:
    if not _HAS_OPENMM:
        logging.warning("OpenMM not available; skipping energy analysis.")
        return None

    system_xml = cfg.get("system_xml") or _get(cfg, "energies.system_xml", None)
    prmtop = cfg.get("prmtop") or _get(cfg, "energies.prmtop", None)
    system = None

    if system_xml:
        logging.info("Loading OpenMM system XML for nonbonded params: %s", system_xml)
        with open(system_xml, "r") as f:
            system = XmlSerializer.deserialize(f.read())
    elif prmtop:
        logging.info("Loading Amber prmtop for nonbonded params: %s", prmtop)
        system = AmberPrmtopFile(str(prmtop)).createSystem(nonbondedMethod=None, constraints=None)
    else:
        logging.warning("No system_xml or prmtop provided; skipping energy analysis.")
        return None

    nb = None
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if isinstance(force, NonbondedForce):
            nb = force
            break
    if nb is None:
        logging.warning("No NonbondedForce found; skipping energy analysis.")
        return None
    if nb.getNumParticles() < traj.n_atoms:
        logging.warning("Nonbonded particle count (%d) < trajectory atoms (%d); skipping energies.", nb.getNumParticles(), traj.n_atoms)
        return None
    if nb.getNumParticles() != traj.n_atoms:
        logging.warning(
            "Nonbonded particle count (%d) != trajectory atoms (%d); using the first %d nonbonded particles. "
            "This is expected when the trajectory was stripped after system creation and atom order was preserved.",
            nb.getNumParticles(),
            traj.n_atoms,
            traj.n_atoms,
        )

    q = np.zeros(traj.n_atoms, dtype=float)
    sig = np.zeros(traj.n_atoms, dtype=float)
    eps = np.zeros(traj.n_atoms, dtype=float)
    for i in range(traj.n_atoms):
        qi, si, ei = nb.getParticleParameters(i)
        q[i] = qi.value_in_unit(unit.elementary_charge)
        sig[i] = si.value_in_unit(unit.nanometer)
        eps[i] = ei.value_in_unit(unit.kilojoule_per_mole)
    return NBParams(q, sig, eps)


def pair_energy_timeseries(traj: md.Trajectory, params: NBParams, idx1: np.ndarray, idx2: np.ndarray, cutoff_nm: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx1 = np.asarray(idx1, dtype=int)
    idx2 = np.asarray(idx2, dtype=int)
    if idx1.size == 0 or idx2.size == 0:
        raise RuntimeError("Cannot compute energy with empty atom selection.")

    coul = np.zeros(traj.n_frames, dtype=float)
    lj = np.zeros(traj.n_frames, dtype=float)
    k_e = 138.935456  # kJ mol-1 nm e-2

    q1 = params.charges_e[idx1]
    q2 = params.charges_e[idx2]
    s1 = params.sigma_nm[idx1]
    s2 = params.sigma_nm[idx2]
    e1 = params.epsilon_kj_mol[idx1]
    e2 = params.epsilon_kj_mol[idx2]

    for iframe in range(traj.n_frames):
        xyz1 = traj.xyz[iframe, idx1, :]
        xyz2 = traj.xyz[iframe, idx2, :]
        d = xyz1[:, None, :] - xyz2[None, :, :]
        r = np.sqrt(np.sum(d * d, axis=2))
        mask = r > 1e-6
        if cutoff_nm is not None:
            mask &= r <= float(cutoff_nm)
        if not np.any(mask):
            continue
        qq = q1[:, None] * q2[None, :]
        sigma = 0.5 * (s1[:, None] + s2[None, :])
        epsilon = np.sqrt(e1[:, None] * e2[None, :])
        rr = np.where(mask, r, 1.0)
        coul_mat = k_e * qq / rr
        sr6 = (sigma / rr) ** 6
        lj_mat = 4.0 * epsilon * (sr6 * sr6 - sr6)
        coul[iframe] = float(np.sum(coul_mat[mask]))
        lj[iframe] = float(np.sum(lj_mat[mask]))
    return coul, lj, coul + lj


def write_energy_rows(out_csv: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "energy_id", "mode", "interaction_type", "frequency", "selection_1", "selection_2",
        "coulomb_mean_kj_mol", "lj_mean_kj_mol", "total_mean_kj_mol",
        "coulomb_std_kj_mol", "lj_std_kj_mol", "total_std_kj_mol", "n_frames",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})
    logging.info("Wrote energy summary: %s", out_csv)


def contact_pair_key(rec: ContactRecord) -> Tuple[str, str]:
    return tuple(sorted((rec.residue_1.label, rec.residue_2.label)))  # type: ignore[return-value]


def energy_summary_row(
    energy_id: str,
    mode: str,
    selection_1: str,
    selection_2: str,
    coul: np.ndarray,
    lj: np.ndarray,
    total: np.ndarray,
    n_frames: int,
    interaction_type: str = "",
    frequency: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "energy_id": energy_id,
        "mode": mode,
        "interaction_type": interaction_type,
        "frequency": "" if frequency is None else f"{frequency:.6f}",
        "selection_1": selection_1,
        "selection_2": selection_2,
        "coulomb_mean_kj_mol": f"{np.mean(coul):.6f}",
        "lj_mean_kj_mol": f"{np.mean(lj):.6f}",
        "total_mean_kj_mol": f"{np.mean(total):.6f}",
        "coulomb_std_kj_mol": f"{np.std(coul):.6f}",
        "lj_std_kj_mol": f"{np.std(lj):.6f}",
        "total_std_kj_mol": f"{np.std(total):.6f}",
        "n_frames": n_frames,
    }


def residue_membership_and_opposite_selection(
    traj: md.Trajectory,
    residue: ResidueKey,
    idx_selection1: np.ndarray,
    selection1_label: str,
    idx_selection2: np.ndarray,
    selection2_label: str,
) -> Tuple[np.ndarray, str]:
    residue_atoms = residue_key_to_atom_indices(traj, residue, heavy_only=False)
    residue_atom_set = set(int(i) for i in residue_atoms)
    selection1_set = set(int(i) for i in idx_selection1)
    selection2_set = set(int(i) for i in idx_selection2)

    in_selection1 = bool(residue_atom_set & selection1_set)
    in_selection2 = bool(residue_atom_set & selection2_set)

    if in_selection1 and not in_selection2:
        idx_opposite = np.setdiff1d(idx_selection2, residue_atoms, assume_unique=False)
        return idx_opposite, selection2_label
    if in_selection2 and not in_selection1:
        idx_opposite = np.setdiff1d(idx_selection1, residue_atoms, assume_unique=False)
        return idx_opposite, selection1_label
    if in_selection1 and in_selection2:
        logging.warning(
            "Residue %s belongs to both selections; using selection2 as opposite and excluding the residue's own atoms.",
            residue.label,
        )
        idx_opposite = np.setdiff1d(idx_selection2, residue_atoms, assume_unique=False)
        return idx_opposite, selection2_label

    raise RuntimeError(
        f"Residue {residue.label} is not part of either configured selection "
        f"({selection1_label!r}, {selection2_label!r})."
    )


def run_energy_analysis(cfg: Dict[str, Any], traj: md.Trajectory, records: List[ContactRecord], outdir: Path) -> None:
    e_cfg = cfg.get("energies", {})
    if not isinstance(e_cfg, dict) or not bool(e_cfg.get("enable", False)):
        return
    params = load_nonbonded_params({**cfg, "energies": e_cfg}, traj)
    if params is None:
        return

    cutoff_nm = e_cfg.get("cutoff_nm", None)
    cutoff_nm = None if cutoff_nm is None else float(cutoff_nm)
    heavy_only = bool(e_cfg.get("heavy_only", False))
    min_frequency = float(e_cfg.get("min_frequency", e_cfg.get("contact_min_frequency", 0.1)))
    energy_records = [rec for rec in records if rec.frequency >= min_frequency]
    logging.info(
        "Energy analysis using %d contact record(s) with frequency >= %.3f.",
        len(energy_records),
        min_frequency,
    )
    rows: List[Dict[str, Any]] = []

    # Mode 1: every contacted residue versus the opposite selection.
    if bool(e_cfg.get("residue_vs_opposite_selection", True)):
        selection1 = str(cfg.get("selection1", "chain A"))
        selection2 = str(cfg.get("selection2", "chain B"))
        idx_selection1 = selection_indices(traj, selection1)
        idx_selection2 = selection_indices(traj, selection2)
        residue_jobs: Dict[Tuple[str, str, str], Tuple[ResidueKey, np.ndarray, str]] = {}

        for rec in energy_records:
            residue_jobs[(rec.residue_1.chain, rec.residue_1.resname, rec.residue_1.resid)] = (rec.residue_1, *residue_membership_and_opposite_selection(
                traj,
                rec.residue_1,
                idx_selection1,
                selection1,
                idx_selection2,
                selection2,
            ))
            residue_jobs[(rec.residue_2.chain, rec.residue_2.resname, rec.residue_2.resid)] = (rec.residue_2, *residue_membership_and_opposite_selection(
                traj,
                rec.residue_2,
                idx_selection1,
                selection1,
                idx_selection2,
                selection2,
            ))

        for i, (_key, (residue, idx_opposite, opposite_label)) in enumerate(sorted(residue_jobs.items()), start=1):
            idx_residue = residue_key_to_atom_indices(traj, residue, heavy_only=heavy_only)
            if idx_residue.size == 0 or idx_opposite.size == 0:
                logging.warning("Skipping residue-vs-selection energy with empty atom match: %s vs %s", residue.label, opposite_label)
                continue
            coul, lj, total = pair_energy_timeseries(traj, params, idx_residue, idx_opposite, cutoff_nm=cutoff_nm)
            rows.append(energy_summary_row(
                f"residue_vs_selection_{i:04d}",
                "residue_vs_opposite_selection",
                residue.label,
                opposite_label,
                coul,
                lj,
                total,
                traj.n_frames,
            ))

    # Mode 2: one deduplicated residue-pair energy per contacted residue pair.
    if bool(e_cfg.get("individual_contacts", False)):
        max_contacts = int(e_cfg.get("max_individual_contacts", 500))
        pair_records: Dict[Tuple[str, str], ContactRecord] = {}
        interaction_types: Dict[Tuple[str, str], set[str]] = defaultdict(set)
        for rec in sorted(energy_records, key=lambda r: r.frequency, reverse=True):
            key = contact_pair_key(rec)
            interaction_types[key].add(rec.interaction_type)
            if key not in pair_records or rec.frequency > pair_records[key].frequency:
                pair_records[key] = rec

        for i, rec in enumerate(list(pair_records.values())[:max_contacts], start=1):
            key = contact_pair_key(rec)
            idx1 = residue_key_to_atom_indices(traj, rec.residue_1, heavy_only=heavy_only)
            idx2 = residue_key_to_atom_indices(traj, rec.residue_2, heavy_only=heavy_only)
            if idx1.size == 0 or idx2.size == 0:
                logging.warning("Skipping energy for contact with empty atom match: %s -- %s", rec.residue_1.label, rec.residue_2.label)
                continue
            coul, lj, total = pair_energy_timeseries(traj, params, idx1, idx2, cutoff_nm=cutoff_nm)
            rows.append(energy_summary_row(
                f"contact_{i:04d}",
                "contact_pair",
                rec.residue_1.label,
                rec.residue_2.label,
                coul,
                lj,
                total,
                traj.n_frames,
                interaction_type=";".join(sorted(interaction_types[key])),
                frequency=rec.frequency,
            ))

    if rows:
        write_energy_rows(outdir / str(e_cfg.get("out_csv", "interaction_energies.csv")), rows)


def map_energies_to_pdb(top_pdb: str | Path, energy_csv: Path, out_pdb: Path, mode: str = "total_abs") -> None:
    if not energy_csv.exists():
        logging.warning("Energy CSV missing; cannot map energies: %s", energy_csv)
        return
    scores: Dict[Tuple[str, str], float] = defaultdict(float)
    with energy_csv.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            val = float(row.get("total_mean_kj_mol", 0.0) or 0.0)
            if mode == "total_abs":
                val = abs(val)
            for col in ("selection_1", "selection_2"):
                tok = row.get(col, "")
                try:
                    key = parse_residue_token(tok)
                except Exception:
                    continue
                scores[(key.chain, key.resid)] += val
    with open(top_pdb, "r", errors="ignore") as fin, out_pdb.open("w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                chain = line[21].strip() or " "
                resseq = line[22:26].strip()
                icode = line[26].strip()
                val = scores.get((chain, resseq + icode), scores.get((chain, resseq), 0.0))
                fout.write(line[:60] + f"{val:6.2f}" + line[66:])
            else:
                fout.write(line)
    logging.info("Wrote energy-mapped PDB: %s", out_pdb)


def write_energy_contacts_pml(
    energy_csv: Path,
    out_pml: Path,
    max_contacts: int = 200,
    attractive_only: bool = True,
    min_width: float = 1.5,
    max_width: float = 10.0,
) -> None:
    if not energy_csv.exists():
        logging.warning("Energy CSV missing; cannot write energy PML: %s", energy_csv)
        return

    rows = []
    with energy_csv.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if row.get("mode") != "contact_pair":
                continue
            try:
                energy = float(row.get("total_mean_kj_mol", 0.0) or 0.0)
            except ValueError:
                continue
            if attractive_only and energy >= 0.0:
                continue
            try:
                r1 = parse_residue_token(row.get("selection_1", ""))
                r2 = parse_residue_token(row.get("selection_2", ""))
            except Exception:
                continue
            rows.append((abs(energy), energy, row, r1, r2))

    rows = sorted(rows, key=lambda item: item[0], reverse=True)[:max_contacts]
    max_abs = max((row[0] for row in rows), default=1.0)

    with out_pml.open("w") as f:
        f.write("# Load your structure first, then run this script in PyMOL.\n")
        f.write("set dash_gap, 0.05\nset dash_radius, 0.08\nhide labels\n")
        if not rows:
            reason = "No attractive contact-pair energies found." if attractive_only else "No contact-pair energies found."
            f.write(f"# {reason}\n")
        for i, (abs_energy, energy, row, r1, r2) in enumerate(rows, start=1):
            n1 = f"en_{i:04d}_a"
            n2 = f"en_{i:04d}_b"
            dname = f"en_{i:04d}"
            color = "tv_blue" if energy < 0 else "firebrick"
            width = min_width + (max_width - min_width) * math.sqrt(abs_energy / max_abs)
            f.write(f"select {n1}, chain {r1.chain} and resi {r1.resid} and name CA+C4*+C4'\n")
            f.write(f"select {n2}, chain {r2.chain} and resi {r2.resid} and name CA+C4*+C4'\n")
            f.write(f"distance {dname}, {n1}, {n2}\n")
            f.write(f"set dash_width, {width:.2f}, {dname}\n")
            f.write(f"color {color}, {dname}\n")
            f.write(f"delete {n1}\ndelete {n2}\n")
        f.write("hide labels\n")
    logging.info("Wrote energy PyMOL contact script: %s", out_pml)
    if not rows:
        logging.warning("Energy PyMOL contact script contains no lines: %s", out_pml)


def run_optional_plots(cfg: Dict[str, Any], outdir: Path, energy_csv: Path) -> None:
    plots_cfg = cfg.get("plots", True)
    if isinstance(plots_cfg, bool):
        if not plots_cfg:
            return
        plots_cfg = {}
    if not isinstance(plots_cfg, dict):
        logging.warning("Ignoring invalid 'plots' config: expected bool or object.")
        return
    if not bool(plots_cfg.get("enable", True)):
        return
    if not energy_csv.exists():
        logging.warning("Energy CSV missing; cannot run interaction summary plots: %s", energy_csv)
        return

    if plots_cfg.get("script"):
        run_external_plot_script(plots_cfg, outdir, energy_csv)
        return

    run_internal_interaction_plots(plots_cfg, outdir, energy_csv)


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "UNK": "X",
}

PLOT_TYPE_COLORS = {
    "sb": "#d62728",
    "hbsb": "#d62728",
    "hbss": "#1f77b4",
    "hbbb": "#1f77b4",
    "hb": "#1f77b4",
    "pc": "#9467bd",
    "ps": "#9467bd",
    "ts": "#8c564b",
    "vdw": "#7f7f7f",
    "wb": "#17becf",
    "lwb": "#17becf",
}

PLOT_CHAIN_COLORS = [
    "#4dc1b7",
    "#25ebd7",
    "#68e679",
    "#b1c361",
    "#16a34a",
    "#dc2626",
    "#0891b2",
    "#ca8a04",
]


def run_external_plot_script(plots_cfg: Dict[str, Any], outdir: Path, energy_csv: Path) -> None:
    script_name = str(plots_cfg["script"])
    candidates = [
        Path(script_name),
        outdir / script_name,
        Path(__file__).resolve().with_name(script_name),
    ]
    script = next((p for p in candidates if p.exists()), None)
    if script is None:
        logging.warning(
            "Plot script %s not found. Checked: %s",
            script_name,
            ", ".join(str(p) for p in candidates),
        )
        return

    prefix = str(plots_cfg.get("prefix", outdir / "interaction_summary"))
    plot_python = str(plots_cfg.get("python", sys.executable))
    cmd = [
        plot_python,
        str(script),
        "-i",
        str(energy_csv),
        "--prefix",
        prefix,
    ]
    cmd.extend(_as_list(plots_cfg.get("extra_args", [])))
    logging.info("Running external interaction summary plots: %s", " ".join(shlex.quote(x) for x in cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        logging.warning("External interaction summary plotting failed with exit code %s; continuing.", exc.returncode)


def run_internal_interaction_plots(plots_cfg: Dict[str, Any], outdir: Path, energy_csv: Path) -> None:
    mpl_cache = outdir / ".matplotlib"
    xdg_cache = outdir / ".cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        logging.warning("Plotting requested but pandas/matplotlib is unavailable: %s", exc)
        return

    prefix = Path(str(plots_cfg.get("prefix", outdir / "interaction_summary")))
    args = argparse.Namespace(
        energy_column=str(plots_cfg.get("energy_column", "total_mean_kj_mol")),
        std_column=str(plots_cfg.get("std_column", "total_std_kj_mol")),
        sort=str(plots_cfg.get("sort", "residue")),
        bar_mode=str(plots_cfg.get("bar_mode", "linear")),
        no_error_bars=bool(plots_cfg.get("no_error_bars", False)),
        normal_ylim=float(plots_cfg.get("normal_ylim", 500.0)),
        fig_width=float(plots_cfg.get("fig_width", 16.0)),
        fig_height=float(plots_cfg.get("fig_height", 8.0)),
        dpi=int(plots_cfg.get("dpi", 300)),
        min_frequency=float(plots_cfg.get("min_frequency", 0.0)),
        top_contacts=plots_cfg.get("top_contacts"),
        include_types=plots_cfg.get("include_types"),
        flare_size=int(plots_cfg.get("flare_size", 1000)),
    )
    if args.sort not in {"residue", "energy", "abs-energy", "input"}:
        logging.warning("Invalid plots.sort=%r; using 'residue'.", args.sort)
        args.sort = "residue"
    if args.bar_mode not in {"broken", "linear", "symlog"}:
        logging.warning("Invalid plots.bar_mode=%r; using 'linear'.", args.bar_mode)
        args.bar_mode = "linear"
    if args.top_contacts is not None:
        args.top_contacts = int(args.top_contacts)

    df = pd.read_csv(energy_csv)
    plot_coerce_numeric(df, pd)

    residue_df = df[df["mode"].astype(str).str.contains("residue", case=False, na=False)]
    contact_df = df[df["mode"].astype(str).eq("contact_pair")]
    if residue_df.empty and contact_df.empty:
        logging.warning("No plottable rows found in %s", energy_csv)
        return
    chain_colors = plot_chain_color_map(
        [
            label
            for label in list(df.get("selection_1", [])) + list(df.get("selection_2", []))
            if plot_is_residue_label(label)
        ]
    )

    residue_png = Path(f"{prefix}_residue_energies.png")
    flare_html = Path(f"{prefix}_flareplot.html")
    if not residue_df.empty:
        plot_residue_bars(residue_df, args, residue_png, plt, chain_colors)
        logging.info("Wrote residue energy plot: %s", residue_png)
    else:
        logging.warning("No residue energy rows found; skipping residue bar plot.")

    if not contact_df.empty:
        flare_data = build_flare_data(contact_df, args, pd, chain_colors)
        write_flareplot(flare_data, args, flare_html)
        logging.info("Wrote interaction flareplot: %s", flare_html)
    else:
        logging.warning("No contact-pair energy rows found; skipping flareplot.")


def plot_residue_sort_key(x: Any) -> Tuple[str, int, str]:
    m = re.match(r"^([^:]+):([^:]+):(-?\d+)([A-Za-z]*)$", str(x))
    if m:
        chain, _resn, num, ins = m.groups()
        return chain, int(num), ins
    return str(x), 0, ""


def plot_is_residue_label(x: Any) -> bool:
    return bool(re.match(r"^([^:]+):([^:]+):(-?\d+[A-Za-z]*)$", str(x)))


def plot_pretty_residue(x: Any) -> str:
    m = re.match(r"^([^:]+):([^:]+):(-?\d+[A-Za-z]*)$", str(x))
    if not m:
        return str(x)
    chain, resn, resid = m.groups()
    aa = AA3_TO_1.get(resn.upper(), resn[0].upper())
    return f"{chain}:{aa}{resid}"


def plot_split_types(x: Any) -> List[str]:
    return [t.strip() for t in str(x).split(";") if t.strip()]


def plot_primary_type(x: Any) -> str:
    priority = ["sb", "hbsb", "hbss", "hbbb", "hb", "pc", "ps", "ts", "vdw", "wb", "lwb"]
    types = plot_split_types(x)
    for p in priority:
        if p in types:
            return p
    return types[0] if types else "contact"


def plot_coerce_numeric(df: Any, pd: Any) -> None:
    for col in [
        "frequency",
        "coulomb_mean_kj_mol", "lj_mean_kj_mol", "total_mean_kj_mol",
        "coulomb_std_kj_mol", "lj_std_kj_mol", "total_std_kj_mol",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def plot_chain_color_map(labels: Iterable[Any]) -> Dict[str, str]:
    chains = sorted({str(label).split(":", 1)[0] for label in labels})
    colors: Dict[str, str] = {}
    for chain in chains:
        colors[chain] = PLOT_CHAIN_COLORS[len(colors) % len(PLOT_CHAIN_COLORS)]
    return colors


def plot_chain_color(label: Any, chain_colors: Dict[str, str]) -> str:
    chain = str(label).split(":", 1)[0]
    return chain_colors.get(chain, "#7f7f7f")


def plot_chain_labels(labels: Iterable[Any], chain_colors: Dict[str, str]) -> List[str]:
    seen = {str(label).split(":", 1)[0] for label in labels}
    return [chain for chain in chain_colors if chain in seen]


def plot_residue_bars(
    residue_df: Any,
    args: argparse.Namespace,
    out_png: Path,
    plt: Any,
    chain_colors: Dict[str, str],
) -> None:
    df = residue_df.copy()
    df = df.dropna(subset=[args.energy_column])
    if df.empty:
        return

    df["label"] = df["selection_1"].map(plot_pretty_residue)
    df["sort_key"] = df["selection_1"].map(plot_residue_sort_key)

    if args.sort == "residue":
        df = df.sort_values("sort_key")
    elif args.sort == "energy":
        df = df.sort_values(args.energy_column)
    elif args.sort == "abs-energy":
        df = df.reindex(df[args.energy_column].abs().sort_values(ascending=False).index)

    x = list(range(len(df)))
    y = df[args.energy_column].to_numpy()
    yerr = None if args.no_error_bars else df[args.std_column].to_numpy()
    colors = chain_colors
    bar_colors = [plot_chain_color(label, colors) for label in df["selection_1"]]

    if args.bar_mode == "broken":
        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(args.fig_width, args.fig_height),
            gridspec_kw={"height_ratios": [1, 3], "hspace": 0.05},
        )
        ax_top.bar(x, y, yerr=yerr, capsize=2, color=bar_colors)
        ax_bot.bar(x, y, yerr=yerr, capsize=2, color=bar_colors)
        normal = args.normal_ylim
        ymax = max(float(np.nanmax(y)) * 1.08, normal * 2)
        ax_bot.set_ylim(-normal, normal)
        ax_top.set_ylim(normal, ymax)
        ax_bot.axhline(0, color="black", linewidth=0.8)
        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)

        d = 0.008
        kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_bot.transAxes)
        ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(df["label"], rotation=90)
        ax_bot.set_xlabel("Residue")
        fig.supylabel(f"{args.energy_column} (kJ/mol)")
        ax_top.set_title("Residue interaction energies")
    else:
        fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
        ax.bar(x, y, yerr=yerr, capsize=2, color=bar_colors)
        ax.axhline(0, color="black", linewidth=0.8)
        if args.bar_mode == "symlog":
            ax.set_yscale("symlog", linthresh=100)
        ax.set_xticks(x)
        ax.set_xticklabels(df["label"], rotation=90)
        ax.set_xlabel("Residue")
        ax.set_ylabel(f"{args.energy_column} (kJ/mol)")
        ax.set_title("Residue interaction energies")
        chains_in_plot = plot_chain_labels(df["selection_1"], colors)
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[chain]) for chain in chains_in_plot]
        ax.legend(handles, [f"chain {chain}" for chain in chains_in_plot], frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=args.dpi)
    plt.close(fig)


def build_flare_data(
    contact_df: Any,
    args: argparse.Namespace,
    pd: Any,
    chain_colors: Dict[str, str],
) -> Dict[str, Any]:
    df = contact_df.copy()
    df = df[df["frequency"].fillna(0) >= args.min_frequency]

    if args.include_types:
        keep = {x.strip() for x in str(args.include_types).split(",") if x.strip()}
        df = df[df["interaction_type"].map(lambda x: bool(set(plot_split_types(x)) & keep))]

    df = df.sort_values("frequency", ascending=False)
    if args.top_contacts:
        df = df.head(args.top_contacts)

    nodes_raw = sorted(
        set(df["selection_1"].astype(str)) | set(df["selection_2"].astype(str)),
        key=plot_residue_sort_key,
    )
    nodes = [
        {
            "id": n,
            "label": plot_pretty_residue(n),
            "chain": n.split(":", 1)[0],
            "chain_color": plot_chain_color(n, chain_colors),
        }
        for n in nodes_raw
    ]

    links = []
    for _, row in df.iterrows():
        primary = plot_primary_type(row["interaction_type"])
        links.append({
            "source": str(row["selection_1"]),
            "target": str(row["selection_2"]),
            "interaction_type": str(row["interaction_type"]),
            "all_types": plot_split_types(row["interaction_type"]),
            "primary_type": primary,
            "frequency": float(row["frequency"]),
            "energy": float(row[args.energy_column]),
            "color": PLOT_TYPE_COLORS.get(primary, "#555555"),
        })

    return {
        "nodes": nodes,
        "links": links,
        "type_colors": PLOT_TYPE_COLORS,
        "chain_colors": chain_colors,
    }


def write_flareplot(data: Dict[str, Any], args: argparse.Namespace, out_html: Path) -> None:
    payload = json.dumps(data)
    size = args.flare_size

    out_html.write_text(f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Interactive flareplot</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
#wrap {{ display: flex; gap: 24px; align-items: flex-start; }}
svg {{ border: 1px solid #ddd; border-radius: 12px; }}
.link {{ fill: none; stroke-opacity: 0.55; cursor: pointer; }}
.link:hover {{ stroke-opacity: 1; }}
.node circle {{ fill: white; stroke: #333; stroke-width: 1.2px; }}
.node text {{ font-size: 12px; }}
.rim {{
    fill: none;
    stroke-width: 18;
    stroke-linecap: round;
    opacity: 1;
}}
.hidden {{ display: none; }}
.swatch {{ display:inline-block; width:24px; height:5px; border-radius:3px; margin-right:8px; }}
.chainbox {{
    display:inline-block;
    width:20px;
    height:10px;
    border-radius:3px;
    margin-right:8px;
    opacity:1;
}}
.row {{ margin: 7px 0; }}
#tooltip {{
  display:none; position:fixed; background:white; border:1px solid #aaa;
  border-radius:8px; padding:8px 10px; box-shadow:0 2px 12px #999;
  pointer-events:none; font-size:13px;
}}
</style>
</head>
<body>
<h2>Interactive flareplot</h2>
<button onclick="setAll(true)">show all</button>
<button onclick="setAll(false)">hide all</button>

<div id="wrap">
<svg id="svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}"></svg>
<div>
<h3>Interaction types</h3>
<div id="controls"></div>
<h3>Chain rim</h3>
<div id="chains"></div>
</div>
</div>
<div id="tooltip"></div>

<script>
const data = {payload};
const svg = document.getElementById("svg");
const tip = document.getElementById("tooltip");
const W = {size}, H = {size}, cx = W/2, cy = H/2;
const r = W * 0.32;
const rimR = r + 95;

const nodeById = new Map(data.nodes.map(d => [d.id, d]));
const allTypes = [...new Set(data.links.flatMap(d => d.all_types))].sort();
const active = new Set(allTypes);
const maxFreq = Math.max(...data.links.map(d => d.frequency || 0), 1e-9);

function make(tag, attrs, parent=svg) {{
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k,v] of Object.entries(attrs || {{}})) el.setAttribute(k,v);
  parent.appendChild(el);
  return el;
}}

function polar(a, rr) {{ return [cx + Math.cos(a)*rr, cy + Math.sin(a)*rr]; }}

function arc(a1,a2,rr) {{
  const p1 = polar(a1,rr), p2 = polar(a2,rr);
  const large = Math.abs(a2-a1) > Math.PI ? 1 : 0;
  return `M ${{p1[0]}} ${{p1[1]}} A ${{rr}} ${{rr}} 0 ${{large}} 1 ${{p2[0]}} ${{p2[1]}}`;
}}

function esc(s) {{
  return String(s).replace(/[&<>"']/g, m => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[m]));
}}

data.nodes.forEach((d,i) => {{
  d.i = i;
  d.angle = -Math.PI/2 + 2*Math.PI*i/data.nodes.length;
  [d.x,d.y] = polar(d.angle,r);
}});

const rimG = make("g");
const linkG = make("g");
const nodeG = make("g");

for (const chain of [...new Set(data.nodes.map(d => d.chain))]) {{
  const ns = data.nodes.filter(d => d.chain === chain);
  const pad = Math.PI / data.nodes.length * 0.65;
  make("path", {{
    d: arc(ns[0].angle - pad, ns[ns.length-1].angle + pad, rimR),
    class: "rim",
    stroke: data.chain_colors[chain],
    "stroke-width": 18,
    "stroke-opacity": 1
  }}, rimG);
}}

for (const l of data.links) {{
  const s = nodeById.get(l.source), t = nodeById.get(l.target);
  const w = 0.8 + 8 * Math.sqrt((l.frequency || 0) / maxFreq);
  const path = make("path", {{
    d: `M ${{s.x}} ${{s.y}} Q ${{cx}} ${{cy}} ${{t.x}} ${{t.y}}`,
    class: "link",
    stroke: l.color,
    "stroke-width": w,
    "data-types": l.all_types.join(",")
  }}, linkG);

  path.onmousemove = e => {{
    tip.style.display = "block";
    tip.style.left = e.clientX + 12 + "px";
    tip.style.top = e.clientY + 12 + "px";
    tip.innerHTML = `<b>${{esc(l.source)}} &harr; ${{esc(l.target)}}</b><br>
      type: ${{esc(l.interaction_type)}}<br>
      frequency: ${{l.frequency.toFixed(3)}}<br>
      energy: ${{l.energy.toFixed(2)}} kJ/mol`;
  }};
  path.onmouseleave = () => tip.style.display = "none";
}}

for (const n of data.nodes) {{
  const deg = n.angle * 180 / Math.PI;
  const flip = deg > 90 || deg < -90;
  const g = make("g", {{
    class: "node",
    transform: `translate(${{n.x}},${{n.y}}) rotate(${{deg}})`
  }}, nodeG);

  make("circle", {{r: 4.5}}, g);

  const text = make("text", {{
    x: flip ? -10 : 10,
    "text-anchor": flip ? "end" : "start",
    transform: flip ? "rotate(180)" : ""
  }}, g);

  text.textContent = n.label;
}}

const controls = document.getElementById("controls");
for (const t of allTypes) {{
  const row = document.createElement("label");
  row.className = "row";
  row.innerHTML = `<input type="checkbox" checked value="${{t}}">
    <span class="swatch" style="background:${{data.type_colors[t] || "#555"}}"></span>${{t}}`;
  row.querySelector("input").onchange = e => {{
    if (e.target.checked) active.add(t);
    else active.delete(t);
    update();
  }};
  controls.appendChild(row);
  controls.appendChild(document.createElement("br"));
}}

const chains = document.getElementById("chains");
for (const [ch,col] of Object.entries(data.chain_colors)) {{
  const row = document.createElement("div");
  row.className = "row";
  row.innerHTML = `<span class="chainbox" style="background:${{col}}"></span>chain ${{esc(ch)}}`;
  chains.appendChild(row);
}}

function update() {{
  document.querySelectorAll(".link").forEach(el => {{
    const ts = el.getAttribute("data-types").split(",");
    el.classList.toggle("hidden", !ts.some(t => active.has(t)));
  }});
}}

function setAll(on) {{
  active.clear();
  if (on) allTypes.forEach(t => active.add(t));
  document.querySelectorAll("#controls input").forEach(cb => cb.checked = on);
  update();
}}
</script>
</body>
</html>
""", encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="getcontacts-based contact analysis + mapping + optional energies.")
    ap.add_argument("-c", "--config", required=True, help="JSON config file")
    ap.add_argument("--loglevel", default="INFO")
    args = ap.parse_args(argv)

    cfg = json.loads(Path(args.config).read_text())
    outdir = Path(cfg.get("output_dir", "contact_analysis_out"))
    setup_logging(outdir, args.loglevel)

    structure = cfg.get("structure") or cfg.get("topology_pdb")
    trajectory = cfg.get("trajectory")
    if not structure:
        raise RuntimeError("Config requires 'structure' or 'topology_pdb'.")

    tsv = run_getcontacts(cfg, outdir)
    if not tsv:
        raise RuntimeError("No getcontacts TSV produced/provided.")

    records_all = parse_getcontacts_tsv(tsv)
    records = filter_contacts(records_all, cfg)
    write_contacts_csv(records, outdir / "contacts_filtered.csv")

    if bool(cfg.get("mapping_contacts", True)):
        write_contact_mapped_pdb(structure, outdir / str(cfg.get("contact_mapped_pdb", "contacts_mapped_bfactor.pdb")), records, scale=float(cfg.get("contact_bfactor_scale", 100.0)))

    pml_cfg = cfg.get("pml", {})
    if isinstance(pml_cfg, dict) and bool(pml_cfg.get("enable", False)):
        write_contacts_pml(
            records,
            outdir / str(pml_cfg.get("out", "contacts.pml")),
            max_contacts=int(pml_cfg.get("max_contacts", 200)),
            min_width=float(pml_cfg.get("min_width", 1.5)),
            max_width=float(pml_cfg.get("max_width", 10.0)),
        )

    # Load trajectory only if energies are requested.
    e_cfg = cfg.get("energies", {})
    if isinstance(e_cfg, dict) and bool(e_cfg.get("enable", False)):
        if not trajectory:
            raise RuntimeError("Energy analysis requires 'trajectory'.")
        stride = int(e_cfg.get("stride", cfg.get("stride", 1)))
        logging.info("Loading trajectory for energies: %s stride=%d", trajectory, stride)
        traj = md.load(str(trajectory), top=str(structure), stride=stride)
        run_energy_analysis(cfg, traj, records, outdir)

        if bool(cfg.get("mapping_energies", False)):
            energy_csv = outdir / str(e_cfg.get("out_csv", "interaction_energies.csv"))
            map_energies_to_pdb(structure, energy_csv, outdir / str(cfg.get("energy_mapped_pdb", "energies_mapped_bfactor.pdb")))
            write_energy_contacts_pml(
                energy_csv,
                outdir / str(cfg.get("energy_contacts_pml", "energy_contacts.pml")),
                max_contacts=int(cfg.get("energy_pml_max_contacts", 200)),
                attractive_only=bool(cfg.get("energy_pml_attractive_only", True)),
                min_width=float(cfg.get("energy_pml_min_width", 1.5)),
                max_width=float(cfg.get("energy_pml_max_width", 10.0)),
            )
            run_optional_plots(cfg, outdir, energy_csv)

    logging.info("Contact analysis complete.")


if __name__ == "__main__":
    main()

