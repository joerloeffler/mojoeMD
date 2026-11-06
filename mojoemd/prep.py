"""System preparation orchestration."""

from __future__ import annotations

from pathlib import Path
import argparse
import logging
import os
import sys

from openmm import XmlSerializer
from openmm.app import ForceField, HBonds, Modeller, PDBFile, PME
from openmm.unit import amu, molar, nanometer

from mojoemd.chemistry.glycans import prepare_glycoprotein_modeller
from mojoemd.chemistry.ligands import _has_openff, parametrize_ligand, prepare_ligand
from mojoemd.chemistry.membranes import choose_membrane_center_z, discard_input_periodic_box
from mojoemd.chemistry.metals import (
    add_metal_coordination_restraints,
    build_metal_aware_protonation_variants,
)
from mojoemd.chemistry.ncaa import (
    load_ncaa_hydrogen_definitions,
    patch_ncaa_internal_bonds_from_ffxml,
    patch_ncaa_peptide_bonds,
)
from mojoemd.chemistry.protein import check_unmatched_residues, prepare_protein, strip_amber_caps
from mojoemd.config import configure_protein_force_field


def run_prepare(args: argparse.Namespace) -> None:
    logging.info("--- STAGE 1: SYSTEM PREPARATION ---")

    configure_protein_force_field(args)
    forcefield_files = [args.protein_ff]
    logging.info(f"Protein force field: {args.protein_ff}")

    if args.membrane:
        logging.info("Membrane mode enabled: adding lipid force field.")
        if args.lipid_ff not in forcefield_files:
            forcefield_files.append(args.lipid_ff)

    forcefield_files.append(args.water_ff)

    if args.glycoprotein:
        logging.info("Glycoprotein mode enabled: adding GLYCAM force field.")
        forcefield_files.append(args.glycam_xml)

    if args.ncaa and args.ncaa_ffxml:
        logging.info("Loading NCAA ffxml files:")
        for ffxml in args.ncaa_ffxml:
            logging.info(f"  - {ffxml}")
            forcefield_files.append(ffxml)

    forcefield = ForceField(*forcefield_files)

    if args.glycoprotein:
        logging.info(f"Loading GLYCAM hydrogen definitions: {args.glycam_hydrogens}")
        Modeller.loadHydrogenDefinitions(args.glycam_hydrogens)

    if args.ncaa and args.ncaa_ffxml:
        load_ncaa_hydrogen_definitions(args.ncaa_ffxml)

    if args.glycoprotein:
        glyco_outdir = Path(args.output_dir) / "glycan_prepare"
        modeller, _glycans_data = prepare_glycoprotein_modeller(args, forcefield, glyco_outdir)
    else:
        protein_fixer = prepare_protein(
            args.receptor,
            ncaa=args.ncaa,
            ncaa_resnames=args.ncaa_resnames,
        )

        modeller = Modeller(protein_fixer.topology, protein_fixer.positions)

        if args.strip_caps:
            strip_amber_caps(modeller)

        if args.ncaa:
            patch_ncaa_internal_bonds_from_ffxml(
                modeller.topology,
                args.ncaa_resnames,
                args.ncaa_ffxml,
            )

            patch_ncaa_peptide_bonds(
                modeller.topology,
                args.ncaa_resnames,
            )

        variants = None
        if args.metal_aware_protonation:
            variants = build_metal_aware_protonation_variants(
                modeller.topology,
                modeller.positions,
                args.metal_coordination_cutoff,
            )

        logging.info("Adding hydrogens using Modeller/ForceField templates...")
        modeller.addHydrogens(forcefield, pH=args.ph, variants=variants)

    ligand_residues_to_ignore = set()

    if args.ligand:
        if not _has_openff:
            logging.error("Cannot process ligand. Please install openff-toolkit and rdkit.")
            sys.exit(1)

        ligand_mol = prepare_ligand(args.ligand, residue_name=args.ligand_resname)
        ligand_topology, ligand_positions = parametrize_ligand(ligand_mol, forcefield, args.lig_ff)
        existing_residue_count = sum(1 for _ in modeller.topology.residues())
        modeller.add(ligand_topology, ligand_positions)
        ligand_residues_to_ignore = {
            residue
            for residue in modeller.topology.residues()
            if residue.index >= existing_residue_count
        }
        for residue in sorted(ligand_residues_to_ignore, key=lambda r: r.index):
            logging.info(
                "Ligand residue will be handled by the registered template generator: "
                f"{residue.name}:{residue.chain.id}:{residue.id}"
            )
        logging.info("Ligand added to the system.")

    check_unmatched_residues(forcefield, modeller.topology, args, ignored_residues=ligand_residues_to_ignore)

    # Solvation and membrane construction create a new periodic box.  Do not
    # inherit placeholder CRYST1 records (PyMOL commonly writes a 1 A cube),
    # since Modeller.addMembrane() otherwise uses that invalid box and its
    # packing relaxation can immediately produce NaN coordinates.
    if discard_input_periodic_box(modeller.topology):
        logging.info("Discarding input periodic box before constructing the solvated system.")

    if args.membrane:
        membrane_center_z = choose_membrane_center_z(modeller, args)

        logging.info(
            "Adding membrane, solvent, and ions "
            f"(lipid={args.lipid_type}, padding={args.membrane_padding:.2f} nm, "
            f"center_z={membrane_center_z:.2f} nm)..."
        )
        modeller.addMembrane(
            forcefield,
            lipidType=args.lipid_type,
            membraneCenterZ=membrane_center_z * nanometer,
            minimumPadding=args.membrane_padding * nanometer,
            positiveIon=args.positive_ion,
            negativeIon=args.negative_ion,
            ionicStrength=args.ionic_strength * molar,
            neutralize=True,
        )
    else:
        logging.info("Adding solvent and ions...")
        modeller.addSolvent(
            forcefield,
            model=args.water_model,
            padding=args.solvent_padding * nanometer,
            positiveIon=args.positive_ion,
            negativeIon=args.negative_ion,
            ionicStrength=args.ionic_strength * molar,
            neutralize=True,
        )
    logging.info(f"System size: {modeller.topology.getNumAtoms()} atoms.")

    logging.info("Creating OpenMM System object...")
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * nanometer,
        constraints=HBonds,
        rigidWater=True,
        hydrogenMass=args.hydrogen_mass * amu,
    )

    if args.metal_coordination_restraints:
        add_metal_coordination_restraints(
            system,
            modeller.topology,
            modeller.positions,
            args.metal_coordination_cutoff,
            args.metal_coordination_k,
        )

    prepared_pdb_path = os.path.join(args.output_dir, "prepared.pdb")
    system_xml_path = os.path.join(args.output_dir, "system.xml")

    logging.info(f"Saving prepared structure to {prepared_pdb_path}")
    with open(prepared_pdb_path, "w") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)

    logging.info(f"Saving serialized System to {system_xml_path}")
    with open(system_xml_path, "w") as f:
        f.write(XmlSerializer.serialize(system))

    logging.info("--- PREPARATION STAGE COMPLETE ---")

