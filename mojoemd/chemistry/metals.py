"""Metal-aware protonation and coordination restraints."""

from __future__ import annotations

import logging
import math

from openmm import CustomBondForce
from openmm.unit import kilojoules_per_mole, nanometer


METAL_ELEMENTS = {
    "LI", "NA", "K", "RB", "CS",
    "BE", "MG", "CA", "SR", "BA",
    "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
    "Y", "ZR", "MO", "RU", "RH", "PD", "AG", "CD",
    "W", "PT", "AU", "HG",
    "AL", "GA", "IN", "SN", "PB",
}


WATER_RESNAMES = {"HOH", "WAT", "H2O", "TIP3", "TIP3P"}


BULK_ION_RESNAMES = {"NA", "CL", "K", "RB", "CS", "LI", "F", "BR", "I"}


COORDINATING_DONOR_ELEMENTS = {"N", "O", "S"}


def atom_element_symbol(atom) -> str:
    if atom.element is not None and atom.element.symbol:
        return atom.element.symbol.upper()

    name = atom.name.strip()
    if name and name[0].isdigit():
        name = name[1:]

    two = name[:2].upper()
    if two in METAL_ELEMENTS or two in {"CL", "BR"}:
        return two

    return name[:1].upper()


def position_distance_nm(pos1, pos2) -> float:
    try:
        p1 = pos1.value_in_unit(nanometer)
    except AttributeError:
        p1 = pos1

    try:
        p2 = pos2.value_in_unit(nanometer)
    except AttributeError:
        p2 = pos2

    return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(3)))


def is_metal_atom(atom) -> bool:
    return atom_element_symbol(atom) in METAL_ELEMENTS


def is_bulk_solvent_or_ion_residue(residue) -> bool:
    return residue.name.upper() in WATER_RESNAMES | BULK_ION_RESNAMES


def build_metal_aware_protonation_variants(
    topology: Topology,
    positions: Quantity,
    cutoff_angstrom: float,
) -> list[str | None]:
    """Force residue protonation variants for common metal-coordinating side chains."""
    residues = list(topology.residues())
    variants: list[str | None] = [None] * len(residues)
    metal_atoms = [atom for atom in topology.atoms() if is_metal_atom(atom)]

    if not metal_atoms:
        logging.info("Metal-aware protonation: no metal atoms detected.")
        return variants

    cutoff_nm = cutoff_angstrom * 0.1
    logging.info(
        "Metal-aware protonation: checking histidine-metal contacts "
        f"within {cutoff_angstrom:.2f} A."
    )

    for residue in residues:
        residue_name = residue.name.upper()

        if residue_name in {"CYS", "CYM", "CYX"}:
            atoms = {atom.name: atom for atom in residue.atoms()}
            sg = atoms.get("SG")

            if sg is None:
                continue

            contacts = []

            for metal in metal_atoms:
                distance = position_distance_nm(positions[sg.index], positions[metal.index])
                if distance <= cutoff_nm:
                    contacts.append((metal, distance))

            if contacts:
                variants[residue.index] = "CYX"
                nearest = min(contacts, key=lambda item: item[1])
                logging.info(
                    "Metal-aware protonation: forcing "
                    f"{residue.name}{residue.id} chain {residue.chain.id} to CYX "
                    f"(SG coordinates {nearest[0].name} at {nearest[1] * 10:.2f} A)."
                )

            continue

        if residue_name not in {"HIS", "HID", "HIE", "HIP"}:
            continue

        atoms = {atom.name: atom for atom in residue.atoms()}
        nd1 = atoms.get("ND1")
        ne2 = atoms.get("NE2")

        if nd1 is None or ne2 is None:
            continue

        nd1_contacts = []
        ne2_contacts = []

        for metal in metal_atoms:
            nd1_distance = position_distance_nm(positions[nd1.index], positions[metal.index])
            ne2_distance = position_distance_nm(positions[ne2.index], positions[metal.index])

            if nd1_distance <= cutoff_nm:
                nd1_contacts.append((metal, nd1_distance))
            if ne2_distance <= cutoff_nm:
                ne2_contacts.append((metal, ne2_distance))

        if nd1_contacts and ne2_contacts:
            logging.warning(
                "Metal-aware protonation: both ND1 and NE2 are close to metal(s) for "
                f"{residue.name}{residue.id} chain {residue.chain.id}; leaving pH-based variant. "
                "This may need an imidazolate/custom template."
            )
            continue

        if nd1_contacts:
            variants[residue.index] = "HIE"
            nearest = min(nd1_contacts, key=lambda item: item[1])
            logging.info(
                "Metal-aware protonation: forcing "
                f"{residue.name}{residue.id} chain {residue.chain.id} to HIE "
                f"(ND1 coordinates {nearest[0].name} at {nearest[1] * 10:.2f} A)."
            )
        elif ne2_contacts:
            variants[residue.index] = "HID"
            nearest = min(ne2_contacts, key=lambda item: item[1])
            logging.info(
                "Metal-aware protonation: forcing "
                f"{residue.name}{residue.id} chain {residue.chain.id} to HID "
                f"(NE2 coordinates {nearest[0].name} at {nearest[1] * 10:.2f} A)."
            )

    return variants


def build_metal_aware_histidine_variants(
    topology: Topology,
    positions: Quantity,
    cutoff_angstrom: float,
) -> list[str | None]:
    """Backward-compatible wrapper for the broader metal-aware protonation logic."""
    return build_metal_aware_protonation_variants(topology, positions, cutoff_angstrom)


def find_metal_coordination_bonds(
    topology: Topology,
    positions: Quantity,
    cutoff_angstrom: float,
) -> list[tuple[int, int, float, str]]:
    cutoff_nm = cutoff_angstrom * 0.1
    atoms = list(topology.atoms())
    metal_atoms = [
        atom
        for atom in atoms
        if is_metal_atom(atom) and atom.residue.name.upper() not in BULK_ION_RESNAMES
    ]
    donor_atoms = [
        atom
        for atom in atoms
        if atom_element_symbol(atom) in COORDINATING_DONOR_ELEMENTS
        and not is_metal_atom(atom)
        and not is_bulk_solvent_or_ion_residue(atom.residue)
    ]

    bonds = []

    for metal in metal_atoms:
        for donor in donor_atoms:
            distance_nm = position_distance_nm(positions[metal.index], positions[donor.index])
            if distance_nm <= cutoff_nm:
                label = (
                    f"{metal.name}:{metal.residue.name}{metal.residue.id}:chain {metal.residue.chain.id} -- "
                    f"{donor.name}:{donor.residue.name}{donor.residue.id}:chain {donor.residue.chain.id}"
                )
                bonds.append((metal.index, donor.index, distance_nm, label))

    return bonds


def add_metal_coordination_restraints(
    system,
    topology: Topology,
    positions: Quantity,
    cutoff_angstrom: float,
    force_constant: float,
) -> None:
    bonds = find_metal_coordination_bonds(topology, positions, cutoff_angstrom)

    if not bonds:
        logging.warning("Metal coordination restraints requested, but no metal-donor contacts were detected.")
        return

    restraint = CustomBondForce("0.5*k_metal_coordination*(r-r0)^2")
    restraint.addGlobalParameter(
        "k_metal_coordination",
        force_constant * kilojoules_per_mole / nanometer**2,
    )
    restraint.addPerBondParameter("r0")

    for metal_index, donor_index, distance_nm, label in bonds:
        restraint.addBond(metal_index, donor_index, [distance_nm])
        logging.info(
            "Metal coordination restraint: "
            f"{label}, r0={distance_nm * 10:.2f} A, k={force_constant:.1f} kJ/mol/nm^2"
        )

    system.addForce(restraint)
    logging.info(f"Added {len(bonds)} metal coordination restraint(s).")

