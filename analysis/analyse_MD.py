#!/usr/bin/env python3
"""
analyse_MD.py

MD trajectory analysis pipeline driven by a JSON config.

Features:
- Optional strip water/ions
- Multichain/complex-aware PBC repair:
    * first makes molecules whole with mdtraj.image_molecules()
    * then shifts each protein chain by whole-box vectors so chains stay close together
    * then centers the repaired complex
- Optional alignment to first frame
- Simple RMSD CSV
- Backbone RMSF / B-factor-like analysis
- SASA analysis
- Optional dihedral angle analysis (phi/psi/omega/chi1-chi5)
- Optional PCA
- Optional hierarchical clustering

Dependencies:
- mdtraj
- numpy
- optional: scikit-learn, scipy

Example:
  python analyse_MD.py -c analysis.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import mdtraj as md
except ImportError as e:
    raise SystemExit("mdtraj is required. Install e.g.: pip install mdtraj") from e

try:
    from sklearn.decomposition import PCA as SKPCA
    from sklearn.cluster import AgglomerativeClustering
    _has_sklearn = True
except Exception:
    _has_sklearn = False

try:
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    from scipy.cluster.hierarchy import fcluster as scipy_fcluster
    _has_scipy = True
except Exception:
    _has_scipy = False


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------


def setup_logging(outdir: str | Path, level: str = "INFO") -> None:
    outdir = str(outdir)
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "analyse_MD.log")

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    logging.info(f"Logging to: {log_path}")


# --------------------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------------------


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _norm_resname(resname: str) -> str:
    s = resname.strip().upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


# --------------------------------------------------------------------------------------
# Chain label recovery from PDB
# --------------------------------------------------------------------------------------


def _parse_pdb_residue_chain_map(pdb_path: str | Path) -> Dict[Tuple[str, int, str], str]:
    """
    Parse PDB ATOM/HETATM lines and build mapping:
      key = (resname, resSeq, iCode) -> chainID
    """
    pdb_path = str(pdb_path)
    mapping: Dict[Tuple[str, int, str], str] = {}
    if not os.path.exists(pdb_path):
        logging.warning(f"Topology PDB not found for chain recovery: {pdb_path}")
        return mapping

    with open(pdb_path, "r", errors="ignore") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                chain_id = line[21]
                resseq_str = line[22:26].strip()
                if not resseq_str:
                    continue
                resseq = int(resseq_str)
                icode = line[26].strip()
                resname = line[17:20].strip().upper()
            except Exception:
                continue

            key = (resname, resseq, icode)
            if key not in mapping:
                mapping[key] = chain_id if chain_id else " "

    logging.info(f"Recovered chain IDs for {len(mapping)} residues from PDB.")
    return mapping


def chain_label_from_mdtraj_residue(
    residue: "md.core.topology.Residue",
    pdb_chain_map: Dict[Tuple[str, int, str], str],
) -> str:
    resname = residue.name.strip().upper()
    resseq = int(residue.resSeq)
    key = (resname, resseq, "")

    if key in pdb_chain_map:
        cid = pdb_chain_map[key].strip()
        if cid:
            return cid
        return "chain" + str(int(residue.chain.index))

    for (rn, rs, _ic), cid in pdb_chain_map.items():
        if rn == resname and rs == resseq:
            cc = cid.strip()
            if cc:
                return cc
            return "chain" + str(int(residue.chain.index))

    cid2 = getattr(residue.chain, "chain_id", None)
    if isinstance(cid2, str) and cid2.strip():
        return cid2.strip()

    return f"chain{int(residue.chain.index)}"


# --------------------------------------------------------------------------------------
# Core operations
# --------------------------------------------------------------------------------------


def load_traj(top_pdb: str | Path, traj_path: str | Path, stride: int = 1) -> md.Trajectory:
    logging.info(f"Loading topology: {top_pdb}")
    logging.info(f"Loading trajectory: {traj_path} (stride={stride})")
    t = md.load(str(traj_path), top=str(top_pdb), stride=int(stride))
    logging.info(f"Loaded: n_frames={t.n_frames}, n_atoms={t.n_atoms}")
    return t


def strip_traj(
    traj: md.Trajectory,
    remove_water: bool,
    remove_ions: bool,
    ion_resnames: List[str],
) -> md.Trajectory:
    if not (remove_water or remove_ions):
        return traj

    ion_set = {_norm_resname(x) for x in ion_resnames}
    water_like = {_norm_resname(x) for x in ["HOH", "WAT", "SOL", "TIP3", "TIP3P", "SPC", "SPCE", "H2O", "OH2", "TIP"]}

    keep: List[int] = []
    for a in traj.topology.atoms:
        rn = _norm_resname(a.residue.name)
        if remove_water and rn in water_like:
            continue
        if remove_ions and rn in ion_set:
            continue
        keep.append(a.index)

    if len(keep) == 0:
        raise RuntimeError("Stripping removed all atoms. Check residue naming and strip settings.")

    out = traj.atom_slice(keep)
    logging.info(
        f"Stripping: remove_water={remove_water}, remove_ions={remove_ions} -> kept {out.n_atoms}/{traj.n_atoms} atoms"
    )
    return out


# --------------------------------------------------------------------------------------
# PBC / imaging / multichain complex repair
# --------------------------------------------------------------------------------------


def _unitcell_vectors_from_lengths_angles(lengths, angles) -> np.ndarray:
    """
    Build box vectors from one frame's unitcell lengths/angles.

    lengths: nm
    angles: degrees
    returns: 3x3 matrix with box vectors as rows
    """
    a, b, c = lengths
    alpha, beta, gamma = np.deg2rad(angles)

    va = np.array([a, 0.0, 0.0], dtype=np.float64)
    vb = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0], dtype=np.float64)

    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(max(c**2 - cx**2 - cy**2, 0.0))

    vc = np.array([cx, cy, cz], dtype=np.float64)
    return np.vstack([va, vb, vc])


def _protein_chain_atom_groups(traj: md.Trajectory, selection: str = "protein") -> Dict[int, np.ndarray]:
    """Return chain_index -> selected atom indices in that chain."""
    idx = traj.topology.select(selection)
    if idx is None or len(idx) == 0:
        raise RuntimeError(f"Selection returned 0 atoms: {selection!r}")

    groups: Dict[int, List[int]] = {}
    for aidx in idx:
        atom = traj.topology.atom(int(aidx))
        cix = int(atom.residue.chain.index)
        groups.setdefault(cix, []).append(int(aidx))

    return {k: np.array(v, dtype=int) for k, v in groups.items()}


def _chain_reference_atom_groups(
    traj: md.Trajectory,
    reference_selection: str = "protein and name CA",
) -> Dict[int, np.ndarray]:
    """Return chain_index -> reference atom indices, usually CA atoms."""
    idx = traj.topology.select(reference_selection)
    if idx is None or len(idx) == 0:
        raise RuntimeError(f"Reference selection returned 0 atoms: {reference_selection!r}")

    groups: Dict[int, List[int]] = {}
    for aidx in idx:
        atom = traj.topology.atom(int(aidx))
        cix = int(atom.residue.chain.index)
        groups.setdefault(cix, []).append(int(aidx))

    return {k: np.array(v, dtype=int) for k, v in groups.items()}


def _residue_atom_groups_from_selection(
    traj: md.Trajectory,
    selection: str,
    exclude_resnames: Optional[List[str]] = None,
) -> Dict[int, np.ndarray]:
    """
    Return residue_index -> atom indices for selected non-protein/small-molecule residues.

    The selection is used to identify candidate atoms, but the whole residue is moved.
    This is usually what you want for a ligand stored as one residue, e.g. LIG/UNK.
    """
    idx = traj.topology.select(selection)
    if idx is None or len(idx) == 0:
        logging.info(f"Small-molecule selection returned 0 atoms: {selection!r}")
        return {}

    exclude = {_norm_resname(x) for x in (exclude_resnames or [])}
    groups: Dict[int, List[int]] = {}
    selected_residues = set()

    for aidx in idx:
        atom = traj.topology.atom(int(aidx))
        residue = atom.residue
        rn = _norm_resname(residue.name)
        if rn in exclude:
            continue
        selected_residues.add(int(residue.index))

    for rix in selected_residues:
        residue = traj.topology.residue(int(rix))
        groups[rix] = [int(a.index) for a in residue.atoms]

    return {k: np.array(v, dtype=int) for k, v in groups.items()}


def repair_small_molecules_pbc(
    traj: md.Trajectory,
    anchor_selection: str = "protein and name CA",
    molecule_selection: str = "not protein and not water",
    exclude_resnames: Optional[List[str]] = None,
) -> md.Trajectory:
    """
    Pull small molecules/ligands into the nearest periodic image around the protein.

    mdtraj.image_molecules() can make molecules whole, but it does not always keep a
    ligand close to the protein after a protein-centered repair. This shifts each
    selected ligand residue by whole box vectors so its center is closest to the
    anchor selection center in every frame.
    """
    if traj.unitcell_lengths is None or traj.unitcell_angles is None:
        logging.info("No unitcell information; skipping small-molecule PBC repair.")
        return traj

    anchor_idx = traj.topology.select(anchor_selection)
    if anchor_idx is None or len(anchor_idx) == 0:
        raise RuntimeError(f"Small-molecule anchor selection returned 0 atoms: {anchor_selection!r}")

    molecule_groups = _residue_atom_groups_from_selection(
        traj,
        selection=molecule_selection,
        exclude_resnames=exclude_resnames,
    )
    if not molecule_groups:
        logging.info("No small molecules selected for ligand-aware PBC repair; skipping.")
        return traj

    xyz = traj.xyz.copy()

    for iframe in range(traj.n_frames):
        box = _unitcell_vectors_from_lengths_angles(traj.unitcell_lengths[iframe], traj.unitcell_angles[iframe])
        inv_box = np.linalg.inv(box)
        anchor_center = xyz[iframe, anchor_idx, :].mean(axis=0)

        for _rix, atom_idx in molecule_groups.items():
            mol_center = xyz[iframe, atom_idx, :].mean(axis=0)
            delta = anchor_center - mol_center
            frac_delta = delta @ inv_box
            shift_frac = np.round(frac_delta)
            shift_cart = shift_frac @ box
            xyz[iframe, atom_idx, :] += shift_cart.astype(np.float32)

    traj.xyz = xyz.astype(np.float32, copy=False)
    logging.info(
        f"Applied ligand-aware PBC repair to {len(molecule_groups)} residue-group(s) "
        f"using molecule_selection={molecule_selection!r}, anchor_selection={anchor_selection!r}"
    )
    return traj


def repair_multichain_complex_pbc(
    traj: md.Trajectory,
    complex_selection: str = "protein",
    reference_atom_selection: str = "protein and name CA",
    anchor_chain_index: Optional[int] = None,
) -> md.Trajectory:
    """
    Chain-wise multichain PBC repair.

    Fixes the common case where one protein chain jumps to the other side of the box.

    Logic:
      1. group protein atoms by chain
      2. choose an anchor chain, default = chain with most reference atoms
      3. for each frame, shift every non-anchor chain by integer box vectors so its
         reference center is closest to the anchor-chain reference center
    """
    if traj.unitcell_lengths is None or traj.unitcell_angles is None:
        logging.info("No unitcell information; skipping multichain PBC repair.")
        return traj

    chain_atoms = _protein_chain_atom_groups(traj, complex_selection)
    chain_refs = _chain_reference_atom_groups(traj, reference_atom_selection)

    common_chains = sorted(set(chain_atoms.keys()) & set(chain_refs.keys()))
    if len(common_chains) < 2:
        logging.info(f"Only {len(common_chains)} chain(s) found for multichain PBC repair; skipping chain-wise repair.")
        return traj

    if anchor_chain_index is None:
        anchor_chain_index = max(common_chains, key=lambda c: len(chain_refs[c]))

    if int(anchor_chain_index) not in common_chains:
        raise RuntimeError(
            f"anchor_chain_index={anchor_chain_index} not found. Available chain indices: {common_chains}"
        )
    anchor_chain_index = int(anchor_chain_index)

    xyz = traj.xyz.copy()

    for iframe in range(traj.n_frames):
        box = _unitcell_vectors_from_lengths_angles(traj.unitcell_lengths[iframe], traj.unitcell_angles[iframe])
        inv_box = np.linalg.inv(box)
        anchor_center = xyz[iframe, chain_refs[anchor_chain_index], :].mean(axis=0)

        for cix in common_chains:
            if cix == anchor_chain_index:
                continue

            chain_center = xyz[iframe, chain_refs[cix], :].mean(axis=0)
            delta = anchor_center - chain_center
            frac_delta = delta @ inv_box
            shift_frac = np.round(frac_delta)
            shift_cart = shift_frac @ box
            xyz[iframe, chain_atoms[cix], :] += shift_cart.astype(np.float32)

    traj.xyz = xyz.astype(np.float32, copy=False)

    logging.info(
        f"Applied chain-wise multichain PBC repair. Anchor chain index: {anchor_chain_index}. "
        f"Chains considered: {common_chains}"
    )
    return traj


def center_complex(traj: md.Trajectory, selection: str = "protein") -> md.Trajectory:
    """Center selected complex in the unit cell."""
    idx = traj.topology.select(selection)
    if idx is None or len(idx) == 0:
        raise RuntimeError(f"Centering selection returned 0 atoms: {selection!r}")

    xyz = traj.xyz.copy()

    for iframe in range(traj.n_frames):
        center = xyz[iframe, idx, :].mean(axis=0)

        if traj.unitcell_lengths is not None and traj.unitcell_angles is not None:
            box = _unitcell_vectors_from_lengths_angles(traj.unitcell_lengths[iframe], traj.unitcell_angles[iframe])
            box_center = 0.5 * box.sum(axis=0)
            shift = box_center - center
        else:
            shift = -center

        xyz[iframe, :, :] += shift.astype(np.float32)

    traj.xyz = xyz.astype(np.float32, copy=False)
    logging.info(f"Centered complex using selection: {selection!r}")
    return traj


def image_and_center(
    traj: md.Trajectory,
    image_molecules: bool = True,
    center: bool = True,
    complex_aware: bool = True,
    complex_selection: str = "protein",
    reference_atom_selection: str = "protein and name CA",
    anchor_chain_index: Optional[int] = None,
    small_molecule_aware: bool = True,
    small_molecule_selection: str = "not protein and not water",
    small_molecule_anchor_selection: str = "protein and name CA",
    small_molecule_exclude_resnames: Optional[List[str]] = None,
) -> md.Trajectory:
    """
    PBC handling.

    Recommended for protein-ligand/multichain complexes:
      image_molecules=True
      complex_aware=True
      small_molecule_aware=True
      center=True
    """
    if traj.unitcell_lengths is None or traj.unitcell_angles is None:
        logging.info("No unitcell information found; skipping imaging/centering.")
        return traj

    if image_molecules:
        try:
            traj.image_molecules(inplace=True)
            logging.info("Applied mdtraj image_molecules.")
        except Exception as e:
            logging.warning(f"mdtraj image_molecules failed; continuing: {e}")

    if complex_aware:
        try:
            traj = repair_multichain_complex_pbc(
                traj,
                complex_selection=complex_selection,
                reference_atom_selection=reference_atom_selection,
                anchor_chain_index=anchor_chain_index,
            )
        except Exception as e:
            logging.warning(f"Multichain PBC repair failed; continuing: {e}")


    if small_molecule_aware:
        try:
            traj = repair_small_molecules_pbc(
                traj,
                anchor_selection=small_molecule_anchor_selection,
                molecule_selection=small_molecule_selection,
                exclude_resnames=small_molecule_exclude_resnames,
            )
        except Exception as e:
            logging.warning(f"Small-molecule PBC repair failed; continuing: {e}")

    if center:
        try:
            traj = center_complex(traj, selection=complex_selection)
        except Exception as e:
            logging.warning(f"Centering failed; continuing: {e}")

    return traj


def align_to_first_frame(traj: md.Trajectory, selection: str = "protein and name CA") -> md.Trajectory:
    idx = traj.topology.select(selection)
    if idx is None or len(idx) == 0:
        raise RuntimeError(f"Alignment selection returned 0 atoms: {selection!r}")
    traj.superpose(traj[0], atom_indices=idx)
    logging.info(f"Aligned trajectory to frame 0 using selection: {selection} (n={len(idx)})")
    return traj


# --------------------------------------------------------------------------------------
# RMSD
# --------------------------------------------------------------------------------------


def compute_rmsd_to_reference(
    traj: md.Trajectory,
    selection: str = "protein and name CA",
    reference_frame: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RMSD to a reference frame in nm.
    Assumes PBC handling and optional alignment have already happened.
    """
    idx = traj.topology.select(selection)
    if idx is None or len(idx) == 0:
        raise RuntimeError(f"RMSD selection returned 0 atoms: {selection!r}")

    reference_frame = int(reference_frame)
    if reference_frame < 0 or reference_frame >= traj.n_frames:
        raise RuntimeError(f"Invalid rmsd.reference_frame={reference_frame}; n_frames={traj.n_frames}")

    rmsd = md.rmsd(traj, traj, frame=reference_frame, atom_indices=idx)
    logging.info(f"Computed RMSD using selection={selection!r}, reference_frame={reference_frame}, n_atoms={len(idx)}")
    return rmsd, idx


def write_rmsd_csv(
    out_csv: str | Path,
    rmsd_nm: np.ndarray,
    dt_ps: Optional[float] = None,
) -> None:
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_ns", "rmsd_nm", "rmsd_A"])
        for i, val in enumerate(rmsd_nm):
            time_ns = "" if dt_ps is None else f"{(i * dt_ps) / 1000.0:.6f}"
            w.writerow([i, time_ns, f"{float(val):.6f}", f"{float(val) * 10.0:.6f}"])
    logging.info(f"Wrote RMSD CSV: {out_csv}")


# --------------------------------------------------------------------------------------
# RMSF / B-factor-like analysis
# --------------------------------------------------------------------------------------


def compute_rmsf(traj: md.Trajectory, atom_indices: np.ndarray) -> np.ndarray:
    """RMSF per atom in nm. Uses aligned coordinates if alignment was enabled."""
    xyz = traj.xyz[:, atom_indices, :]
    mean = xyz.mean(axis=0, keepdims=True)
    dev2 = ((xyz - mean) ** 2).sum(axis=2)
    return np.sqrt(dev2.mean(axis=0))


def residue_rmsf_from_atom_rmsf(traj: md.Trajectory, atom_indices: np.ndarray, atom_rmsf: np.ndarray) -> Dict[int, float]:
    res_to_vals: Dict[int, List[float]] = {}
    sel_set = set(int(i) for i in atom_indices)
    atom_rmsf_map = {int(atom_indices[i]): float(atom_rmsf[i]) for i in range(len(atom_indices))}

    for a in traj.topology.atoms:
        if a.index not in sel_set:
            continue
        rix = a.residue.index
        res_to_vals.setdefault(rix, []).append(atom_rmsf_map[a.index])

    return {rix: float(np.mean(vals)) for rix, vals in res_to_vals.items()}


def write_atom_rmsf_csv(
    traj: md.Trajectory,
    atom_indices: np.ndarray,
    atom_rmsf: np.ndarray,
    out_csv: str | Path,
    pdb_chain_map: Dict[Tuple[str, int, str], str],
) -> None:
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["atom_index", "chain", "chain_index", "resSeq", "resName", "atomName", "rmsf_nm"])
        for i, aidx in enumerate(atom_indices):
            a = traj.topology.atom(int(aidx))
            r = a.residue
            chain_str = chain_label_from_mdtraj_residue(r, pdb_chain_map)
            w.writerow([a.index, chain_str, int(r.chain.index), r.resSeq, r.name, a.name, f"{float(atom_rmsf[i]):.6f}"])
    logging.info(f"Wrote per-atom RMSF CSV: {out_csv}")


def write_residue_rmsf_csv(
    traj: md.Trajectory,
    res_rmsf: Dict[int, float],
    out_csv: str | Path,
    pdb_chain_map: Dict[Tuple[str, int, str], str],
) -> None:
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chain", "chain_index", "resSeq", "resName", "residue_index", "rmsf_nm"])
        for rix in sorted(res_rmsf.keys()):
            r = traj.topology.residue(int(rix))
            chain_str = chain_label_from_mdtraj_residue(r, pdb_chain_map)
            w.writerow([chain_str, int(r.chain.index), r.resSeq, r.name, r.index, f"{res_rmsf[rix]:.6f}"])
    logging.info(f"Wrote per-residue RMSF CSV: {out_csv}")


def write_pdb_with_bfactors(
    traj: md.Trajectory,
    bfactor_mode: str,
    atom_indices: np.ndarray,
    atom_rmsf: np.ndarray,
    res_rmsf: Dict[int, float],
    out_pdb: str | Path,
) -> None:
    b = np.zeros(traj.n_atoms, dtype=np.float32)

    if bfactor_mode == "atom":
        for i, aidx in enumerate(atom_indices):
            b[int(aidx)] = float(atom_rmsf[i])
    elif bfactor_mode == "residue":
        for rix, val in res_rmsf.items():
            r = traj.topology.residue(int(rix))
            for a in r.atoms:
                b[a.index] = float(val)
    else:
        raise ValueError("bfactor_mode must be 'atom' or 'residue'")

    traj[0].save_pdb(str(out_pdb), bfactors=b)
    logging.info(f"Wrote PDB with B-factors={bfactor_mode} RMSF: {out_pdb}")


# --------------------------------------------------------------------------------------
# SASA
# --------------------------------------------------------------------------------------


def sasa_analysis(
    traj: md.Trajectory,
    probe_radius_nm: float = 0.14,
    mode: str = "residue",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if mode not in ("atom", "residue"):
        raise ValueError("sasa.mode must be 'atom' or 'residue'")

    sr = md.shrake_rupley(traj, probe_radius=float(probe_radius_nm), mode=mode)
    global_sasa = sr.sum(axis=1)
    if mode == "atom":
        return global_sasa, None
    return global_sasa, sr


def write_global_sasa_csv(out_csv: str | Path, global_sasa_nm2: np.ndarray, dt_ps: Optional[float] = None) -> None:
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_ns", "sasa_nm2"])
        for i, v in enumerate(global_sasa_nm2):
            time_ns = "" if dt_ps is None else f"{(i * dt_ps) / 1000.0:.6f}"
            w.writerow([i, time_ns, f"{float(v):.6f}"])
    logging.info(f"Wrote global SASA CSV: {out_csv}")


def write_residue_sasa_timeseries_csv(
    traj: md.Trajectory,
    out_csv: str | Path,
    sasa_by_residue: np.ndarray,
    dt_ps: Optional[float],
    pdb_chain_map: Dict[Tuple[str, int, str], str],
) -> None:
    labels = [f"{chain_label_from_mdtraj_residue(r, pdb_chain_map)}:{r.resSeq}:{r.name}" for r in traj.topology.residues]

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_ns"] + labels)
        for i in range(sasa_by_residue.shape[0]):
            time_ns = "" if dt_ps is None else f"{(i * dt_ps) / 1000.0:.6f}"
            row = [i, time_ns] + [f"{float(x):.6f}" for x in sasa_by_residue[i]]
            w.writerow(row)
    logging.info(f"Wrote residue-wise SASA timeseries CSV: {out_csv}")


def write_residue_sasa_mean_csv(
    traj: md.Trajectory,
    out_csv: str | Path,
    sasa_by_residue: np.ndarray,
    pdb_chain_map: Dict[Tuple[str, int, str], str],
) -> None:
    mean = sasa_by_residue.mean(axis=0)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chain", "chain_index", "resSeq", "resName", "residue_index", "mean_sasa_nm2"])
        for i, r in enumerate(traj.topology.residues):
            w.writerow([
                chain_label_from_mdtraj_residue(r, pdb_chain_map),
                int(r.chain.index),
                r.resSeq,
                r.name,
                r.index,
                f"{float(mean[i]):.6f}",
            ])
    logging.info(f"Wrote mean residue-wise SASA CSV: {out_csv}")



# --------------------------------------------------------------------------------------
# Dihedral angle analysis
# --------------------------------------------------------------------------------------


def _angle_deg_signed(rad: np.ndarray) -> np.ndarray:
    """Convert radians to wrapped signed degrees in [-180, 180)."""
    deg = np.rad2deg(rad)
    return ((deg + 180.0) % 360.0) - 180.0


def _circular_mean_deg(rad_vals: np.ndarray) -> float:
    """Circular mean for angles in radians, returned as signed degrees."""
    if rad_vals.size == 0:
        return float("nan")
    s = np.nanmean(np.sin(rad_vals))
    c = np.nanmean(np.cos(rad_vals))
    return float(_angle_deg_signed(np.array([np.arctan2(s, c)]))[0])


def _circular_std_deg(rad_vals: np.ndarray) -> float:
    """
    Circular standard deviation for angles in radians, returned in degrees.
    Formula: sqrt(-2 ln R), where R is mean resultant length.
    """
    if rad_vals.size == 0:
        return float("nan")
    s = np.nanmean(np.sin(rad_vals))
    c = np.nanmean(np.cos(rad_vals))
    R = float(np.sqrt(s * s + c * c))
    R = max(min(R, 1.0), 1e-12)
    return float(np.rad2deg(np.sqrt(-2.0 * np.log(R))))


def _dihedral_label_from_atoms(
    traj: md.Trajectory,
    atom_quad: np.ndarray,
    dtype: str,
    pdb_chain_map: Dict[Tuple[str, int, str], str],
) -> str:
    """
    Create a stable human-readable label for one dihedral.

    For backbone angles we label the biologically central residue:
      phi   = residue of atom 2 in the quartet, N(i)
      psi   = residue of atom 1 in the quartet, N(i)
      omega = peptide bond residue i -> i+1
    For side-chain chi angles, all/most atoms belong to the same residue; use atom 1.
    """
    dtype = dtype.lower()
    atom_quad = np.asarray(atom_quad, dtype=int)
    atoms = [traj.topology.atom(int(i)) for i in atom_quad]

    if dtype == "phi":
        r = atoms[1].residue
        prefix = "phi"
    elif dtype == "psi":
        r = atoms[0].residue
        prefix = "psi"
    elif dtype == "omega":
        r1 = atoms[1].residue
        r2 = atoms[2].residue
        c1 = chain_label_from_mdtraj_residue(r1, pdb_chain_map)
        c2 = chain_label_from_mdtraj_residue(r2, pdb_chain_map)
        return f"omega:{c1}:{r1.resSeq}:{r1.name}-{c2}:{r2.resSeq}:{r2.name}"
    else:
        r = atoms[1].residue
        prefix = dtype

    chain = chain_label_from_mdtraj_residue(r, pdb_chain_map)
    return f"{prefix}:{chain}:{r.resSeq}:{r.name}"


def _compute_dihedral_type(traj: md.Trajectory, dtype: str) -> Tuple[np.ndarray, np.ndarray]:
    dtype = dtype.lower()
    funcs = {
        "phi": md.compute_phi,
        "psi": md.compute_psi,
        "omega": md.compute_omega,
        "chi1": md.compute_chi1,
        "chi2": md.compute_chi2,
        "chi3": md.compute_chi3,
        "chi4": md.compute_chi4,
        "chi5": md.compute_chi5,
    }
    if dtype not in funcs:
        raise ValueError(
            f"Unsupported dihedral type {dtype!r}. Use any of: {', '.join(sorted(funcs))}"
        )
    atom_indices, angles_rad = funcs[dtype](traj)
    return np.asarray(atom_indices, dtype=int), np.asarray(angles_rad, dtype=np.float64)


def write_dihedral_timeseries_csv(
    out_csv: str | Path,
    labels: List[str],
    angles_rad: np.ndarray,
    dt_ps: Optional[float] = None,
    units: str = "degrees",
) -> None:
    units = units.lower()
    if units not in ("degrees", "radians"):
        raise ValueError("dihedrals.units must be 'degrees' or 'radians'")

    values = _angle_deg_signed(angles_rad) if units == "degrees" else angles_rad
    suffix = "deg" if units == "degrees" else "rad"

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_ns"] + [f"{x}_{suffix}" for x in labels])
        for iframe in range(values.shape[0]):
            time_ns = "" if dt_ps is None else f"{(iframe * dt_ps) / 1000.0:.6f}"
            w.writerow([iframe, time_ns] + [f"{float(x):.6f}" for x in values[iframe]])
    logging.info(f"Wrote dihedral timeseries CSV: {out_csv}")


def append_dihedral_summary_rows(
    rows: List[List[Any]],
    traj: md.Trajectory,
    dtype: str,
    atom_indices: np.ndarray,
    labels: List[str],
    angles_rad: np.ndarray,
    pdb_chain_map: Dict[Tuple[str, int, str], str],
) -> None:
    for j, label in enumerate(labels):
        quad = atom_indices[j]
        atoms = [traj.topology.atom(int(i)) for i in quad]
        res_labels = []
        atom_labels = []
        for a in atoms:
            r = a.residue
            ch = chain_label_from_mdtraj_residue(r, pdb_chain_map)
            res_labels.append(f"{ch}:{r.resSeq}:{r.name}")
            atom_labels.append(f"{ch}:{r.resSeq}:{r.name}:{a.name}")

        vals_rad = angles_rad[:, j]
        vals_deg = _angle_deg_signed(vals_rad)
        rows.append([
            dtype,
            label,
            ";".join(res_labels),
            ";".join(atom_labels),
            _circular_mean_deg(vals_rad),
            _circular_std_deg(vals_rad),
            float(np.nanmin(vals_deg)),
            float(np.nanmax(vals_deg)),
            int(np.sum(~np.isnan(vals_rad))),
        ])


def run_dihedral_analysis(
    traj: md.Trajectory,
    outdir: Path,
    cfg: Dict[str, Any],
    dt_ps: Optional[float],
    pdb_chain_map: Dict[Tuple[str, int, str], str],
) -> None:
    """
    Compute backbone/side-chain dihedral angles and write CSV outputs.

    JSON block example:
      "dihedrals": {
        "enable": true,
        "types": ["phi", "psi", "omega", "chi1"],
        "units": "degrees",
        "write_timeseries_csv": true,
        "write_summary_csv": true,
        "out_prefix": "dihedrals"
      }
    """
    types = cfg.get("types", ["phi", "psi"])
    if isinstance(types, str):
        types = [x.strip() for x in types.split(",") if x.strip()]
    types = [str(x).lower() for x in types]

    units = str(cfg.get("units", "degrees")).lower()
    out_prefix = str(cfg.get("out_prefix", "dihedrals"))
    write_ts = bool(cfg.get("write_timeseries_csv", True))
    write_summary = bool(cfg.get("write_summary_csv", True))

    summary_rows: List[List[Any]] = []

    for dtype in types:
        atom_idx, angles_rad = _compute_dihedral_type(traj, dtype)
        if atom_idx.size == 0 or angles_rad.size == 0:
            logging.warning(f"No {dtype} dihedrals found; skipping.")
            continue

        labels = [
            _dihedral_label_from_atoms(traj, atom_idx[j], dtype, pdb_chain_map)
            for j in range(atom_idx.shape[0])
        ]

        if write_ts:
            out_csv = outdir / f"{out_prefix}_{dtype}_timeseries.csv"
            write_dihedral_timeseries_csv(out_csv, labels, angles_rad, dt_ps=dt_ps, units=units)

        if write_summary:
            append_dihedral_summary_rows(
                summary_rows,
                traj,
                dtype,
                atom_idx,
                labels,
                angles_rad,
                pdb_chain_map,
            )

        logging.info(f"Computed {dtype} dihedrals: n={atom_idx.shape[0]}")

    if write_summary and summary_rows:
        out_csv = outdir / f"{out_prefix}_summary.csv"
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "dihedral_type",
                "label",
                "residues",
                "atoms",
                "circular_mean_deg",
                "circular_std_deg",
                "min_deg",
                "max_deg",
                "n_frames",
            ])
            for row in summary_rows:
                row2 = row[:4] + [f"{float(x):.6f}" if isinstance(x, float) else x for x in row[4:]]
                w.writerow(row2)
        logging.info(f"Wrote dihedral summary CSV: {out_csv}")


# --------------------------------------------------------------------------------------
# PCA and clustering
# --------------------------------------------------------------------------------------


def _selection_indices(traj: md.Trajectory, selection: str) -> np.ndarray:
    idx = traj.topology.select(selection)
    if idx is None or len(idx) == 0:
        raise RuntimeError(f"Selection returned 0 atoms: {selection!r}")
    return idx


def _coords_matrix(traj: md.Trajectory, atom_indices: np.ndarray, subsample: int = 1) -> np.ndarray:
    subsample = max(1, int(subsample))
    t = traj[::subsample]
    xyz = t.xyz[:, atom_indices, :]
    return xyz.reshape(xyz.shape[0], -1).astype(np.float32, copy=False)


def run_pca(
    traj: md.Trajectory,
    outdir: Path,
    selection: str = "protein and backbone",
    n_components: int = 10,
    subsample: int = 1,
) -> Dict[str, Any]:
    atom_idx = _selection_indices(traj, selection)
    X = _coords_matrix(traj, atom_idx, subsample=subsample)
    Xc = X - X.mean(axis=0, keepdims=True)

    if _has_sklearn:
        pca = SKPCA(n_components=int(n_components), svd_solver="randomized", random_state=0)
        proj = pca.fit_transform(Xc)
        evr = pca.explained_variance_ratio_
        evals = pca.explained_variance_
    else:
        U, S, _Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(int(n_components), len(S))
        proj = (U[:, :k] * S[:k]).astype(np.float32, copy=False)
        var = (S**2) / (Xc.shape[0] - 1)
        evals = var[:k]
        evr = (var[:k] / var.sum()).astype(np.float64, copy=False)

    eig_path = outdir / "pca_eigenvalues.csv"
    with open(eig_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pc", "eigenvalue"])
        for i, v in enumerate(np.atleast_1d(evals), start=1):
            w.writerow([i, float(v)])
    logging.info(f"Wrote PCA eigenvalues: {eig_path}")

    evr_path = outdir / "pca_explained_variance.csv"
    with open(evr_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pc", "explained_variance_ratio"])
        for i, v in enumerate(np.atleast_1d(evr), start=1):
            w.writerow([i, float(v)])
    logging.info(f"Wrote PCA explained variance: {evr_path}")

    proj_path = outdir / "pca_projection.csv"
    with open(proj_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["frame_sub"] + [f"PC{i}" for i in range(1, proj.shape[1] + 1)]
        w.writerow(header)
        for i in range(proj.shape[0]):
            w.writerow([i] + [f"{float(x):.6f}" for x in proj[i]])
    logging.info(f"Wrote PCA projection: {proj_path}")

    return {"proj": proj, "subsample": int(subsample), "explained_variance_ratio": np.asarray(evr)}


def run_hierarchical_clustering_on_pca(
    proj: np.ndarray,
    outdir: Path,
    cutoff: float = 2.0,
    linkage_method: str = "average",
) -> np.ndarray:
    cutoff = float(cutoff)
    linkage_method = str(linkage_method).lower()

    if _has_scipy:
        Z = scipy_linkage(proj, method=linkage_method, metric="euclidean")
        labels = scipy_fcluster(Z, t=cutoff, criterion="distance")
    elif _has_sklearn:
        model = AgglomerativeClustering(n_clusters=None, distance_threshold=cutoff, linkage=linkage_method)
        labels = model.fit_predict(proj) + 1
    else:
        raise ImportError("Need scipy or scikit-learn for clustering.")

    sizes_path = outdir / "cluster_sizes.csv"
    uniq, cnt = np.unique(labels, return_counts=True)
    with open(sizes_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "n_frames"])
        for c, n in zip(uniq, cnt):
            w.writerow([int(c), int(n)])
    logging.info(f"Wrote cluster sizes: {sizes_path}")

    return labels.astype(int)


def write_cluster_assignments(
    traj: md.Trajectory,
    labels_sub: np.ndarray,
    out_csv: Path,
    subsample: int,
    dt_ps: Optional[float],
) -> None:
    subsample = max(1, int(subsample))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_ns", "cluster_id"])
        for i_sub, lab in enumerate(labels_sub):
            frame = i_sub * subsample
            time_ns = "" if dt_ps is None else f"{(frame * dt_ps) / 1000.0:.6f}"
            w.writerow([frame, time_ns, int(lab)])
    logging.info(f"Wrote cluster assignments: {out_csv}")


def write_cluster_representatives_pdb(
    traj: md.Trajectory,
    proj: np.ndarray,
    labels: np.ndarray,
    outdir: Path,
    subsample: int,
    max_reps: int = 20,
) -> None:
    subsample = max(1, int(subsample))
    max_reps = int(max_reps)
    reps_dir = outdir / "cluster_representatives"
    reps_dir.mkdir(parents=True, exist_ok=True)

    clusters = np.unique(labels)
    if len(clusters) > max_reps:
        logging.warning(f"Too many clusters ({len(clusters)}). Limiting representatives to first {max_reps} clusters.")
        clusters = clusters[:max_reps]

    for c in clusters:
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        mu = proj[idx].mean(axis=0, keepdims=True)
        d2 = ((proj[idx] - mu) ** 2).sum(axis=1)
        best_sub = idx[int(np.argmin(d2))]
        best_frame = int(best_sub * subsample)
        out_pdb = reps_dir / f"cluster_{int(c):03d}_frame_{best_frame}.pdb"
        traj[best_frame].save_pdb(str(out_pdb))
    logging.info(f"Wrote cluster representative PDBs to: {reps_dir}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="MD analysis pipeline with multichain PBC repair.")
    ap.add_argument("-c", "--config", required=True, help="Path to analysis JSON config.")
    ap.add_argument("--loglevel", default="INFO", help="Logging level: INFO, DEBUG, WARNING.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())

    outdir = Path(cfg.get("output_dir", "analysis_out"))
    setup_logging(outdir, level=args.loglevel)

    top_pdb = cfg["topology_pdb"]
    traj_path = cfg["trajectory"]
    stride = int(cfg.get("stride", 1))

    pdb_chain_map = _parse_pdb_residue_chain_map(top_pdb)
    traj = load_traj(top_pdb=top_pdb, traj_path=traj_path, stride=stride)

    dt_ps = _get(cfg, "time.dt_ps", None)
    dt_ps = float(dt_ps) if dt_ps is not None else None

    traj = strip_traj(
        traj,
        remove_water=bool(_get(cfg, "strip.remove_water", False)),
        remove_ions=bool(_get(cfg, "strip.remove_ions", False)),
        ion_resnames=list(_get(cfg, "strip.ion_resnames", ["NA", "CL", "K", "MG", "CA", "ZN"])),
    )

    if bool(_get(cfg, "imaging.enable", True)):
        anchor_raw = _get(cfg, "imaging.anchor_chain_index", None)
        anchor_chain_index = None if anchor_raw is None else int(anchor_raw)
        traj = image_and_center(
            traj,
            image_molecules=bool(_get(cfg, "imaging.image_molecules", True)),
            center=bool(_get(cfg, "imaging.center", True)),
            complex_aware=bool(_get(cfg, "imaging.complex_aware", True)),
            complex_selection=str(_get(cfg, "imaging.complex_selection", "protein")),
            reference_atom_selection=str(_get(cfg, "imaging.reference_atom_selection", "protein and name CA")),
            anchor_chain_index=anchor_chain_index,
            small_molecule_aware=bool(_get(cfg, "imaging.small_molecule_aware", True)),
            small_molecule_selection=str(_get(cfg, "imaging.small_molecule_selection", "not protein and not water")),
            small_molecule_anchor_selection=str(_get(cfg, "imaging.small_molecule_anchor_selection", "protein and name CA")),
            small_molecule_exclude_resnames=list(_get(cfg, "imaging.small_molecule_exclude_resnames", ["HOH", "WAT", "SOL", "TIP3", "TIP3P", "SPC", "SPCE", "H2O", "OH2", "TIP", "NA", "CL", "K", "MG", "CA", "ZN", "CU", "MN", "FE", "CO"])),
        )

    if bool(_get(cfg, "alignment.enable", True)):
        sel = str(_get(cfg, "alignment.selection", "protein and name CA"))
        traj = align_to_first_frame(traj, selection=sel)

    # RMSD after PBC repair and optional alignment
    if bool(_get(cfg, "rmsd.enable", True)):
        rmsd_sel = str(_get(cfg, "rmsd.selection", "protein and name CA"))
        rmsd_ref = int(_get(cfg, "rmsd.reference_frame", 0))
        rmsd_nm, _rmsd_idx = compute_rmsd_to_reference(traj, selection=rmsd_sel, reference_frame=rmsd_ref)
        out_csv = outdir / str(_get(cfg, "rmsd.out_csv", "rmsd.csv"))
        write_rmsd_csv(out_csv, rmsd_nm, dt_ps=dt_ps)

    if bool(cfg.get("write_processed", True)):
        proc_top = outdir / "processed_topology.pdb"
        proc_dcd = outdir / "processed.dcd"
        traj[0].save_pdb(str(proc_top))
        traj.save_dcd(str(proc_dcd))
        logging.info(f"Wrote processed topology: {proc_top}")
        logging.info(f"Wrote processed trajectory: {proc_dcd}")

    if bool(_get(cfg, "bfactor_rmsf.enable", True)):
        atom_sel = str(_get(cfg, "bfactor_rmsf.atom_selection", "protein and backbone"))
        atom_idx = traj.topology.select(atom_sel)
        if atom_idx is None or len(atom_idx) == 0:
            raise RuntimeError(f"bfactor_rmsf.atom_selection returned 0 atoms: {atom_sel!r}")

        atom_rmsf = compute_rmsf(traj, atom_idx)
        write_atom_rmsf_csv(traj, atom_idx, atom_rmsf, outdir / "rmsf_atoms.csv", pdb_chain_map)

        res_rmsf = residue_rmsf_from_atom_rmsf(traj, atom_idx, atom_rmsf)
        write_residue_rmsf_csv(traj, res_rmsf, outdir / "rmsf_residues.csv", pdb_chain_map)

        if bool(_get(cfg, "bfactor_rmsf.write_pdb_with_bfactor", True)):
            out_pdb = outdir / str(_get(cfg, "bfactor_rmsf.bfactor_pdb_out", "bfactor_rmsf.pdb"))
            per_res = bool(_get(cfg, "bfactor_rmsf.per_residue", True))
            mode = "residue" if per_res else "atom"
            write_pdb_with_bfactors(traj, mode, atom_idx, atom_rmsf, res_rmsf, out_pdb)

    if bool(_get(cfg, "sasa.enable", True)):
        probe = float(_get(cfg, "sasa.probe_radius_nm", 0.14))
        mode = str(_get(cfg, "sasa.mode", "residue"))
        global_sasa, sasa_by_res = sasa_analysis(traj, probe_radius_nm=probe, mode=mode)

        if bool(_get(cfg, "sasa.write_global_csv", True)):
            write_global_sasa_csv(outdir / "sasa_global.csv", global_sasa, dt_ps=dt_ps)

        if sasa_by_res is not None:
            if bool(_get(cfg, "sasa.write_residuewise_timeseries_csv", True)):
                write_residue_sasa_timeseries_csv(
                    traj,
                    outdir / "sasa_residue_timeseries.csv",
                    sasa_by_res,
                    dt_ps=dt_ps,
                    pdb_chain_map=pdb_chain_map,
                )
            if bool(_get(cfg, "sasa.write_residuewise_mean_csv", True)):
                write_residue_sasa_mean_csv(
                    traj,
                    outdir / "sasa_residue_mean.csv",
                    sasa_by_res,
                    pdb_chain_map=pdb_chain_map,
                )

    dih_cfg = cfg.get("dihedrals", {})
    if isinstance(dih_cfg, dict) and bool(dih_cfg.get("enable", False)):
        run_dihedral_analysis(
            traj,
            outdir,
            dih_cfg,
            dt_ps=dt_ps,
            pdb_chain_map=pdb_chain_map,
        )

    pca_cfg = cfg.get("pca", {})
    pca_result = None
    if isinstance(pca_cfg, dict) and bool(pca_cfg.get("enable", False)):
        pca_result = run_pca(
            traj,
            outdir,
            selection=str(pca_cfg.get("selection", "protein and backbone")),
            n_components=int(pca_cfg.get("n_components", 10)),
            subsample=int(pca_cfg.get("subsample", 1)),
        )

    cl_cfg = cfg.get("clustering", {})
    if isinstance(cl_cfg, dict) and bool(cl_cfg.get("enable", False)):
        cl_sel = str(cl_cfg.get("selection", "protein and backbone"))
        cl_sub = int(cl_cfg.get("subsample", 1))
        linkage_method = str(cl_cfg.get("linkage", "average"))
        cutoff = float(cl_cfg.get("cutoff", 2.0))
        use_pca_space = bool(cl_cfg.get("use_pca_space", True))

        if use_pca_space:
            if pca_result is None:
                n_comp = max(int(cl_cfg.get("pca_components_for_clustering", 5)), 2)
                pca_result = run_pca(traj, outdir, selection=cl_sel, n_components=max(n_comp, 10), subsample=cl_sub)
            proj = pca_result["proj"]
            k = int(cl_cfg.get("pca_components_for_clustering", min(5, proj.shape[1])))
            k = max(2, min(k, proj.shape[1]))
            feat = proj[:, :k]
            subsample_used = int(pca_result.get("subsample", cl_sub))
        else:
            atom_idx = _selection_indices(traj, cl_sel)
            feat = _coords_matrix(traj, atom_idx, subsample=cl_sub)
            subsample_used = cl_sub

        labels = run_hierarchical_clustering_on_pca(feat, outdir, cutoff=cutoff, linkage_method=linkage_method)
        write_cluster_assignments(traj, labels, outdir / "clusters.csv", subsample=subsample_used, dt_ps=dt_ps)

        if bool(cl_cfg.get("write_representatives", True)):
            write_cluster_representatives_pdb(
                traj,
                feat,
                labels,
                outdir,
                subsample=subsample_used,
                max_reps=int(cl_cfg.get("max_representatives", 20)),
            )

    logging.info("Analysis complete.")


if __name__ == "__main__":
    main()