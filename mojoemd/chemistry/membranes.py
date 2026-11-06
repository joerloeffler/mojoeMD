"""Membrane placement and input periodic-box handling."""

from __future__ import annotations

from collections import defaultdict
import argparse
import logging

from openmm.unit import nanometer


def parse_int_residue_id(residue_id: str) -> int | None:
    try:
        return int(residue_id)
    except (TypeError, ValueError):
        return None


KYTE_DOOLITTLE_HYDROPATHY = {
    "ALA": 1.8,
    "ARG": -4.5,
    "ASN": -3.5,
    "ASP": -3.5,
    "CYS": 2.5,
    "GLN": -3.5,
    "GLU": -3.5,
    "GLY": -0.4,
    "HIS": -3.2,
    "ILE": 4.5,
    "LEU": 3.8,
    "LYS": -3.9,
    "MET": 1.9,
    "PHE": 2.8,
    "PRO": -1.6,
    "SER": -0.8,
    "THR": -0.7,
    "TRP": -0.9,
    "TYR": -1.3,
    "VAL": 4.2,
}


def infer_membrane_center_z_from_hydrophobic_slab(
    modeller: Modeller,
    thickness_nm: float,
    window_size: int = 21,
) -> float | None:
    positions_nm = modeller.positions.value_in_unit(nanometer)
    residues_by_chain: dict[str, list[tuple[int, str, float]]] = defaultdict(list)

    for atom in modeller.topology.atoms():
        residue = atom.residue
        residue_name = residue.name.upper()
        if residue_name not in KYTE_DOOLITTLE_HYDROPATHY:
            continue
        if atom.name != "CA":
            continue

        residue_id = parse_int_residue_id(residue.id)
        if residue_id is None:
            continue

        residues_by_chain[residue.chain.id].append((
            residue_id,
            residue_name,
            positions_nm[atom.index][2],
        ))

    if not residues_by_chain:
        return None

    best = None
    min_z_span_nm = min(1.4, thickness_nm * 0.5)

    for chain_id, residues in residues_by_chain.items():
        residues = sorted(residues)
        if len(residues) < window_size:
            continue

        for start in range(len(residues) - window_size + 1):
            window = residues[start:start + window_size]
            first_resid = window[0][0]
            last_resid = window[-1][0]

            if last_resid - first_resid > window_size + 5:
                continue

            z_values = [item[2] for item in window]
            z_span = max(z_values) - min(z_values)
            if z_span < min_z_span_nm:
                continue

            hydropathy = sum(KYTE_DOOLITTLE_HYDROPATHY[item[1]] for item in window) / window_size
            if hydropathy < 1.0:
                continue

            score = hydropathy * z_span
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "chain_id": chain_id,
                    "first_resid": first_resid,
                    "last_resid": last_resid,
                    "hydropathy": hydropathy,
                    "z_span": z_span,
                    "center_z": 0.5 * (min(z_values) + max(z_values)),
                }

    if best is None:
        return None

    logging.info(
        "Auto-selected membrane center z from hydrophobic TM-window heuristic: "
        f"{best['center_z']:.2f} nm "
        f"(chain={best['chain_id']}, residues={best['first_resid']}-{best['last_resid']}, "
        f"hydropathy={best['hydropathy']:.2f}, z_span={best['z_span']:.2f} nm)"
    )
    return best["center_z"]


def choose_membrane_center_z(modeller: Modeller, args: argparse.Namespace) -> float:
    if args.membrane_center_z is not None:
        return args.membrane_center_z

    selected_positions = []
    positions_nm = modeller.positions.value_in_unit(nanometer)
    residue_range = args.membrane_center_residue_range
    chain_ids = set(args.membrane_center_chain_ids or [])

    if residue_range or chain_ids:
        for atom in modeller.topology.atoms():
            residue = atom.residue

            if chain_ids and residue.chain.id not in chain_ids:
                continue

            if residue_range:
                residue_id = parse_int_residue_id(residue.id)
                if residue_id is None:
                    continue
                if residue_id < residue_range[0] or residue_id > residue_range[1]:
                    continue

            selected_positions.append(positions_nm[atom.index])

        if not selected_positions:
            raise ValueError(
                "No atoms matched the membrane-centering selection. "
                "Check --membrane_center_chain_ids and --membrane_center_residue_range."
            )

        z_values = [position[2] for position in selected_positions]
        membrane_center_z = 0.5 * (min(z_values) + max(z_values))
        selection = []
        if chain_ids:
            selection.append("chains=" + ",".join(sorted(chain_ids)))
        if residue_range:
            selection.append(f"residues={residue_range[0]}-{residue_range[1]}")
        logging.info(
            "Auto-selected membrane center z from "
            f"{' '.join(selection)}: {membrane_center_z:.2f} nm"
        )
        return membrane_center_z

    membrane_center_z = infer_membrane_center_z_from_hydrophobic_slab(
        modeller,
        args.membrane_auto_thickness,
    )
    if membrane_center_z is not None:
        return membrane_center_z

    z_values = [position[2] for position in positions_nm]
    membrane_center_z = 0.5 * (min(z_values) + max(z_values))
    logging.warning(
        "Could not infer a hydrophobic membrane slab; falling back to all prepared atoms "
        f"for membrane center z: {membrane_center_z:.2f} nm"
    )

    return membrane_center_z


def discard_input_periodic_box(topology: Topology) -> bool:
    """Clear an inherited box before constructing a new solvated system."""
    if topology.getPeriodicBoxVectors() is None:
        return False
    topology.setPeriodicBoxVectors(None)
    return True

