"""Production dynamics and checkpoint continuation."""

from __future__ import annotations

from pathlib import Path
import argparse
import logging
import os
import re
import sys

from openmm import LangevinMiddleIntegrator, XmlSerializer
from openmm.app import CheckpointReporter, DCDReporter, PDBFile, Simulation, StateDataReporter
from openmm.unit import femtoseconds, kelvin, picosecond, picoseconds

from mojoemd.runtime import LoggingReporter, select_platform


def run_production(args: argparse.Namespace) -> None:
    logging.info("--- STAGE 3: PRODUCTION MD ---")

    equil_system_path = os.path.join(args.output_dir, "equilibrated_system.xml")
    equil_pdb_path = os.path.join(args.output_dir, "equilibrated.pdb")
    equil_state_path = os.path.join(args.output_dir, "equilibrated.xml")

    checkpoint_path = os.path.join(args.output_dir, "production.chk")

    if not all(os.path.exists(p) for p in [equil_system_path, equil_pdb_path]):
        logging.error("Cannot find equilibrated files. Please run --prepare and --equilibrate first.")
        sys.exit(1)

    logging.info(f"Loading topology from {equil_pdb_path}")
    pdb = PDBFile(equil_pdb_path)
    topology = pdb.topology

    logging.info(f"Loading system from {equil_system_path}")
    with open(equil_system_path, "r") as f:
        system = XmlSerializer.deserialize(f.read())

    platform = select_platform()
    integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, args.timestep_fs * femtoseconds)
    simulation = Simulation(topology, system, integrator, platform)

    if args.restart:
        restart_candidates = [
            *Path(args.output_dir).glob("production_final_*.chk"),
            *(
                Path(path)
                for path in (
                    os.path.join(args.output_dir, "production_final.chk"),
                    checkpoint_path,
                )
                if os.path.exists(path)
            ),
        ]

        if restart_candidates:
            latest_checkpoint = max(restart_candidates, key=os.path.getmtime)
            logging.info(f"Restart requested. Loading checkpoint from {latest_checkpoint}")
            simulation.loadCheckpoint(str(latest_checkpoint))
        else:
            logging.error("--restart was requested, but no checkpoint file was found.")
            sys.exit(1)

    else:
        if not os.path.exists(equil_state_path):
            logging.error("Cannot find equilibrated state. Please run --equilibrate first, or use --restart.")
            sys.exit(1)

        logging.info(f"Loading state from {equil_state_path}")
        simulation.loadState(equil_state_path)

    segment = next_production_segment(args.output_dir)
    dcd_path = os.path.join(args.output_dir, f"production_{segment:03d}.dcd")
    log_path = os.path.join(args.output_dir, f"production_{segment:03d}.csv")
    final_checkpoint_path = os.path.join(
        args.output_dir,
        f"production_final_{segment:03d}.chk",
    )
    logging.info(
        "Writing production segment %03d to %s and %s",
        segment,
        dcd_path,
        log_path,
    )

    dt_ps = integrator.getStepSize().value_in_unit(picoseconds)
    report_interval_ps = 50
    report_interval = int(report_interval_ps / dt_ps)
    production_steps = int(args.prod_ns * 1000 / dt_ps)

    checkpoint_interval = max(1, int(args.checkpoint_interval_ps / dt_ps))

    total_steps_for_reporter = simulation.currentStep + production_steps

    simulation.reporters.append(DCDReporter(dcd_path, report_interval))
    simulation.reporters.append(
        StateDataReporter(
            log_path,
            report_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            density=True,
            progress=True,
            remainingTime=True,
            speed=True,
            totalSteps=total_steps_for_reporter,
            separator=",",
        )
    )
    simulation.reporters.append(CheckpointReporter(checkpoint_path, checkpoint_interval))
    simulation.reporters.append(LoggingReporter())

    logging.info(f"Starting {args.prod_ns} ns production run ({production_steps} steps)...")
    simulation.step(production_steps)

    logging.info(f"Saving final checkpoint to {final_checkpoint_path}")
    simulation.saveCheckpoint(final_checkpoint_path)

    logging.info("--- PRODUCTION STAGE COMPLETE ---")


def next_production_segment(output_dir: str | Path) -> int:
    """Return the next unused numbered production trajectory segment."""
    patterns = (
        re.compile(r"^production_(\d+)\.(?:dcd|csv)$"),
        re.compile(r"^production_final_(\d+)\.chk$"),
    )
    segments = []
    for path in Path(output_dir).glob("production_*.*"):
        for pattern in patterns:
            match = pattern.match(path.name)
            if match:
                segments.append(int(match.group(1)))
                break
    return max(segments, default=-1) + 1

