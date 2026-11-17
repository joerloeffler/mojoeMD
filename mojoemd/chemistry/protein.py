"""Protein repair, cap removal, and force-field template checks."""

from __future__ import annotations

import argparse
import logging
import sys

from openmm.app import NoCutoff
import pdbfixer


def strip_amber_caps(modeller: Modeller) -> None:
    cap_residues = [
        residue
        for residue in modeller.topology.residues()
        if residue.name.upper() in {"ACE", "NME"}
    ]

    if not cap_residues:
        logging.info("Terminal cap stripping requested, but no ACE/NME residues were found.")
        return

    labels = [f"{residue.name} {residue.chain.id}:{residue.id}" for residue in cap_residues]
    modeller.delete(cap_residues)
    logging.info("Removed terminal caps: " + ", ".join(labels))


def prepare_protein(
    pdb_file: str | Path,
    ncaa: bool = False,
    ncaa_resnames: list[str] | None = None,
) -> pdbfixer.PDBFixer:
    logging.info(f"Preparing protein from {pdb_file}...")

    fixer = pdbfixer.PDBFixer(str(pdb_file))

    fixer.findMissingResidues()
    fixer.findNonstandardResidues()

    ncaa_resnames_set = set(ncaa_resnames or [])

    if ncaa:
        logging.info("NCAA mode enabled.")

        if ncaa_resnames_set:
            logging.info(f"Preserving NCAA residue names: {', '.join(sorted(ncaa_resnames_set))}")
        else:
            logging.warning(
                "NCAA mode enabled but no --ncaa_resnames were provided. "
                "All nonstandard residues will be preserved."
            )

        kept = []
        replace = []

        for residue, suggested_replacement in fixer.nonstandardResidues:
            if not ncaa_resnames_set or residue.name in ncaa_resnames_set:
                kept.append(residue.name)
            else:
                replace.append((residue, suggested_replacement))

        fixer.nonstandardResidues = replace

        if replace:
            logging.info(
                "Replacing nonstandard residues not marked as NCAA: "
                + ", ".join(sorted({res.name for res, _ in replace}))
            )
            fixer.replaceNonstandardResidues()

        if kept:
            logging.info("Kept NCAA/nonstandard residues unchanged: " + ", ".join(sorted(set(kept))))

    else:
        fixer.replaceNonstandardResidues()

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    logging.info("Skipping PDBFixer hydrogen addition; hydrogens will be added with Modeller/ForceField.")
    logging.info("Protein preparation complete.")
    return fixer


def check_unmatched_residues(
    forcefield: ForceField,
    topology: Topology,
    args: argparse.Namespace,
    ignored_residues: set | None = None,
) -> None:
    if args.protein_ff.startswith("charmm"):
        # getUnmatchedResidues only matches base templates, omitting CHARMM
        # terminal and multi-residue patches. createSystem applies those patches
        # and checks parameter completeness without creating a simulation context.
        forcefield.createSystem(topology, nonbondedMethod=NoCutoff)
        logging.info("All residues matched by CHARMM templates and patches.")
        return
    ignored_residues = ignored_residues or set()
    unmatched = [
        residue
        for residue in forcefield.getUnmatchedResidues(topology)
        if residue not in ignored_residues
    ]

    if not unmatched:
        logging.info("All residues matched by the force field.")
        return

    unmatched_names = sorted({res.name for res in unmatched})
    logging.warning("Unmatched residues before createSystem/addSolvent: " + ", ".join(unmatched_names))

    if args.ncaa:
        ncaa_set = set(args.ncaa_resnames or [])
        missing_ncaa = [res for res in unmatched if not ncaa_set or res.name in ncaa_set]

        if missing_ncaa:
            logging.error(
                "Some NCAA residues still have no matching force-field template: "
                + ", ".join(sorted({res.name for res in missing_ncaa}))
            )
            logging.error(
                "For embedded NCAAs, provide or fix a residue-template ffxml via "
                "--ncaa_ffxml ncaa/ALM.xml"
            )
            sys.exit(1)

    logging.error(
        "Unmatched residues found. Use --ncaa with --ncaa_resnames and --ncaa_ffxml "
        "if these are intended noncanonical residues."
    )
    sys.exit(1)

