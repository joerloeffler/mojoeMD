"""Noncanonical amino-acid bonds and hydrogen definitions."""

from __future__ import annotations

from pathlib import Path
import logging
import sys
import xml.etree.ElementTree as ET

from openmm.app import Modeller


def patch_ncaa_internal_bonds_from_ffxml(
    topology: Topology,
    ncaa_resnames: list[str],
    ncaa_ffxml_files: list[str],
) -> None:
    """Patch missing intra-residue NCAA bonds using Bond records from ffxml.

    This is needed because OpenMM/PDBFixer often does not infer internal bonds
    for unknown residue names such as ALM.
    """
    ncaa_set = set(ncaa_resnames or [])

    if not ncaa_set:
        logging.warning("NCAA mode enabled but no residue names given; skipping NCAA internal-bond patching.")
        return

    if not ncaa_ffxml_files:
        logging.warning("No NCAA ffxml files provided; skipping NCAA internal-bond patching.")
        return

    residue_bonds: dict[str, list[tuple[str, str]]] = {}

    for ffxml in ncaa_ffxml_files:
        ffxml_path = Path(ffxml)

        if not ffxml_path.exists():
            logging.error(f"NCAA ffxml file does not exist: {ffxml_path}")
            sys.exit(1)

        tree = ET.parse(ffxml_path)
        root = tree.getroot()

        for residue_elem in root.findall(".//Residue"):
            resname = residue_elem.attrib.get("name")

            if resname not in ncaa_set:
                continue

            bonds = []

            for bond_elem in residue_elem.findall("Bond"):
                a1 = bond_elem.attrib.get("atomName1")
                a2 = bond_elem.attrib.get("atomName2")

                if a1 and a2:
                    bonds.append((a1, a2))

            residue_bonds[resname] = bonds
            logging.info(f"Read {len(bonds)} internal bond definition(s) for NCAA residue {resname} from {ffxml_path}")

    existing_bonds = {frozenset((bond[0].index, bond[1].index)) for bond in topology.bonds()}
    added = 0

    for residue in topology.residues():
        if residue.name not in ncaa_set:
            continue

        atoms = {atom.name: atom for atom in residue.atoms()}

        if residue.name not in residue_bonds:
            logging.warning(f"No internal bond definitions found for NCAA residue {residue.name}")
            continue

        for atom1_name, atom2_name in residue_bonds[residue.name]:
            if atom1_name not in atoms or atom2_name not in atoms:
                continue

            atom1 = atoms[atom1_name]
            atom2 = atoms[atom2_name]
            bond_key = frozenset((atom1.index, atom2.index))

            if bond_key not in existing_bonds:
                topology.addBond(atom1, atom2)
                existing_bonds.add(bond_key)
                added += 1
                logging.info(
                    f"Added NCAA internal bond: "
                    f"{residue.name}{residue.id}:{atom1_name} -- {atom2_name}"
                )

    logging.info(f"Patched {added} NCAA internal bond(s) from ffxml.")


def patch_ncaa_peptide_bonds(topology: Topology, ncaa_resnames: list[str]) -> None:
    """Patch missing peptide bonds around NCAA residues.

    Adds:
      previous residue C -- NCAA N
      NCAA C -- next residue N

    when either residue in an adjacent pair is an NCAA.
    """
    ncaa_set = set(ncaa_resnames or [])

    if not ncaa_set:
        logging.warning("NCAA mode enabled but no residue names given; skipping peptide-bond patching.")
        return

    existing_bonds = {frozenset((bond[0].index, bond[1].index)) for bond in topology.bonds()}

    def atom_by_name(residue, atom_name: str):
        for atom in residue.atoms():
            if atom.name == atom_name:
                return atom
        return None

    added = 0

    for chain in topology.chains():
        residues = list(chain.residues())

        for prev_res, next_res in zip(residues[:-1], residues[1:]):
            if prev_res.name not in ncaa_set and next_res.name not in ncaa_set:
                continue

            c_atom = atom_by_name(prev_res, "C")
            n_atom = atom_by_name(next_res, "N")

            if c_atom is None or n_atom is None:
                logging.warning(
                    f"Could not patch peptide bond {prev_res.name}{prev_res.id} -> "
                    f"{next_res.name}{next_res.id}: missing C or N atom."
                )
                continue

            bond_key = frozenset((c_atom.index, n_atom.index))

            if bond_key not in existing_bonds:
                topology.addBond(c_atom, n_atom)
                existing_bonds.add(bond_key)
                added += 1
                logging.info(
                    f"Added peptide bond: {prev_res.name}{prev_res.id}:C -- "
                    f"{next_res.name}{next_res.id}:N"
                )

    logging.info(f"Patched {added} NCAA-adjacent peptide bond(s).")


def load_ncaa_hydrogen_definitions(ncaa_ffxml_files: list[str]) -> None:
    """Load optional hydrogen-definition files next to NCAA ffxml files.

    Example:
      ncaa/ALM.xml -> ncaa/ALM_hydrogens.xml
    """
    for ffxml in ncaa_ffxml_files:
        ffxml_path = Path(ffxml)
        hxml_path = ffxml_path.with_name(ffxml_path.stem + "_hydrogens.xml")

        if hxml_path.exists():
            logging.info(f"Loading NCAA hydrogen definitions: {hxml_path}")
            Modeller.loadHydrogenDefinitions(str(hxml_path))
        else:
            logging.warning(f"No NCAA hydrogen definition file found: {hxml_path}")

