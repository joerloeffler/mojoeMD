"""Equilibration of prepared systems."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from openmm import CustomExternalForce, LangevinMiddleIntegrator, MonteCarloBarostat, XmlSerializer
from openmm.app import PDBFile, Simulation
from openmm.unit import bar, femtoseconds, kelvin, kilojoules_per_mole, nanometer, picosecond
from tqdm import tqdm

from mojoemd.runtime import select_platform


def run_equilibrate(args: argparse.Namespace) -> None:
    logging.info("--- STAGE 2: SYSTEM EQUILIBRATION ---")

    prepared_pdb_path = os.path.join(args.output_dir, "prepared.pdb")
    system_xml_path = os.path.join(args.output_dir, "system.xml")

    if not all(os.path.exists(p) for p in [prepared_pdb_path, system_xml_path]):
        logging.error("Cannot find prepared files. Please run the --prepare step first.")
        sys.exit(1)

    logging.info(f"Loading topology from {prepared_pdb_path}")
    pdb = PDBFile(prepared_pdb_path)
    topology = pdb.topology
    positions = pdb.positions

    logging.info(f"Loading system from {system_xml_path}")
    with open(system_xml_path, "r") as f:
        system = XmlSerializer.deserialize(f.read())

    platform = select_platform()
    integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, args.timestep_fs * femtoseconds)
    simulation = Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)

    logging.info("Running energy minimization...")
    simulation.minimizeEnergy()
    logging.info("Minimization complete.")

    logging.info("Heating system to 300K (NVT)...")
    simulation.context.setVelocitiesToTemperature(5 * kelvin)

    for temp in tqdm(range(5, 305, 5), desc="NVT Heating", ncols=80):
        integrator.setTemperature(temp * kelvin)
        simulation.step(500)

    integrator.setTemperature(300 * kelvin)
    logging.info("Heating complete.")

    logging.info("Equilibrating system (NPT) with decreasing restraints...")
    system.addForce(MonteCarloBarostat(1 * bar, 300 * kelvin))

    restraint = CustomExternalForce("0.5*k_backbone_restraint*periodicdistance(x,y,z,x0,y0,z0)^2")
    restraint.addGlobalParameter("k_backbone_restraint", 4000.0 * kilojoules_per_mole / nanometer**2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    backbone_indices = [
        a.index
        for a in topology.atoms()
        if a.residue.name not in ("HOH", "NA", "CL") and a.name in ("CA", "C", "N", "O")
    ]

    logging.info(f"Applying position restraints to {len(backbone_indices)} backbone atoms.")

    initial_positions = simulation.context.getState(getPositions=True).getPositions()

    for i in backbone_indices:
        restraint.addParticle(i, initial_positions[i])

    system.addForce(restraint)
    simulation.context.reinitialize(preserveState=True)

    restraint_levels = [4000.0, 2000.0, 1000.0, 500.0, 250.0, 100.0, 50.0, 0.0]

    for k in tqdm(restraint_levels, desc="NPT Equilibration", ncols=80):
        simulation.context.setParameter("k_backbone_restraint", k)
        simulation.step(12500)

    logging.info("NPT equilibration complete.")

    equil_system_path = os.path.join(args.output_dir, "equilibrated_system.xml")
    equil_state_path = os.path.join(args.output_dir, "equilibrated.xml")
    equil_pdb_path = os.path.join(args.output_dir, "equilibrated.pdb")

    logging.info(f"Saving equilibrated system with barostat and restraints to {equil_system_path}")
    with open(equil_system_path, "w") as f:
        f.write(XmlSerializer.serialize(system))

    logging.info(f"Saving equilibrated state to {equil_state_path}")
    simulation.saveState(equil_state_path)

    logging.info(f"Saving equilibrated structure to {equil_pdb_path}")
    final_positions = simulation.context.getState(getPositions=True).getPositions()
    with open(equil_pdb_path, "w") as f:
        PDBFile.writeFile(topology, final_positions, f, keepIds=True)

    logging.info("--- EQUILIBRATION STAGE COMPLETE ---")

