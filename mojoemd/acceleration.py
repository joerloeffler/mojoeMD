"""Optional cuPotential production dynamics; stock preparation stays portable."""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import logging
import math
from pathlib import Path
import platform

import openmm as mm
from openmm import unit


class AccelerationError(RuntimeError):
    """Requested acceleration is unavailable or incompatible with this run."""


def load_cupotential():
    """Load only on explicit request, including the plugin's native libraries."""
    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise AccelerationError("cuPotential requires Linux x86-64 and an NVIDIA CUDA GPU.")
    if not _version("openmm").startswith("8.4."):
        raise AccelerationError(
            "This cuPotential integration targets OpenMM 8.4.x, matching the nv-openmm-plugin "
            "0.3.5 binary requirements. Use a separate compatible cluster environment."
        )
    if _version("nv-openmm-plugin") != "0.3.5":
        raise AccelerationError(
            "Install nv-openmm-plugin 0.3.5 in the active cluster environment; "
            "other early-access plugin versions are not covered by this adapter."
        )
    try:
        mm.Platform.getPlatformByName("CUDA")
        plugin = importlib.import_module("nv_openmm_plugin")
        # Import alone can succeed even when the plugin's native libraries fail.
        for name in ("NonbondedForce", "LangevinMiddleIntegrator", "MonteCarloBarostat"):
            getattr(plugin, name)
        return plugin
    except Exception as exc:
        raise AccelerationError(
            "cuPotential could not load. Install nv-openmm-plugin built for this OpenMM/"
            f"Python/CUDA environment and run 'mojoemd doctor --acceleration cupotential': {exc}"
        ) from exc


def validate_system(system: mm.System) -> None:
    """Conservative first integration: rectangular PME and standard Amber forces."""
    box = system.getDefaultPeriodicBoxVectors()
    values = [v.value_in_unit(unit.nanometer) for v in box]
    if (any(not math.isfinite(x) for row in values for x in row)
            or any(values[i][i] <= 0 for i in range(3))
            or any(abs(values[i][j]) > 1e-10 for i in range(3) for j in range(3) if i != j)):
        raise AccelerationError("cuPotential integration requires a rectangular periodic box.")
    allowed = (mm.NonbondedForce, mm.HarmonicBondForce, mm.HarmonicAngleForce,
               mm.PeriodicTorsionForce, mm.RBTorsionForce, mm.CMMotionRemover,
               mm.MonteCarloBarostat, mm.CustomBondForce, mm.CustomExternalForce)
    nonbonded = []
    barostats = []
    removers = []
    for force in system.getForces():
        if not isinstance(force, allowed):
            raise AccelerationError(
                f"cuPotential integration does not yet support {type(force).__name__}; "
                "use --acceleration openmm for this system."
            )
        if isinstance(force, mm.NonbondedForce):
            nonbonded.append(force)
            if force.getNonbondedMethod() != mm.NonbondedForce.PME:
                raise AccelerationError("cuPotential integration currently requires PME (not LJPME).")
            if force.getNumParticleParameterOffsets() or force.getNumExceptionParameterOffsets():
                raise AccelerationError("cuPotential integration does not support nonbonded parameter offsets.")
            if force.getEwaldErrorTolerance() != 5e-4:
                raise AccelerationError("cuPotential currently requires an Ewald error tolerance of 5e-4.")
            alpha, nx, ny, nz = force.getPMEParameters()
            if alpha.value_in_unit(unit.nanometer**-1) != 0 or any((nx, ny, nz)):
                raise AccelerationError("cuPotential integration requires automatic PME parameters.")
            if force.getReciprocalSpaceForceGroup() not in (-1, force.getForceGroup()):
                raise AccelerationError("cuPotential integration requires a shared direct/reciprocal force group.")
            cutoff = force.getCutoffDistance().value_in_unit(unit.nanometer)
            if min(values[i][i] for i in range(3)) <= 2 * cutoff:
                raise AccelerationError("The periodic box must exceed twice the nonbonded cutoff.")
        elif isinstance(force, mm.MonteCarloBarostat):
            barostats.append(force)
        elif isinstance(force, mm.CMMotionRemover):
            removers.append(force)
    if len(nonbonded) != 1 or len(barostats) > 1 or len(removers) > 1:
        raise AccelerationError("cuPotential requires one NonbondedForce and at most one barostat/COM remover.")


def cupotential_dynamics(system: mm.System, timestep_fs: float):
    """Convert a clone before Context creation; preserve settings and other forces."""
    validate_system(system)
    plugin = load_cupotential()
    converted = mm.XmlSerializer.clone(system)
    removals = []
    com_frequency = 0
    for index, force in enumerate(converted.getForces()):
        replacement = None
        if isinstance(force, mm.NonbondedForce):
            replacement = plugin.NonbondedForce(force)
        elif isinstance(force, mm.MonteCarloBarostat):
            replacement = plugin.MonteCarloBarostat(
                force.getDefaultPressure(), force.getDefaultTemperature(), force.getFrequency()
            )
            replacement.setRandomNumberSeed(force.getRandomNumberSeed())
        elif isinstance(force, mm.CMMotionRemover):
            com_frequency = force.getFrequency()
            removals.append(index)
        if replacement is not None:
            replacement.setName(force.getName())
            replacement.setForceGroup(force.getForceGroup())
            converted.addForce(replacement)
            removals.append(index)
    for index in sorted(removals, reverse=True):
        converted.removeForce(index)
    integrator = plugin.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, timestep_fs * unit.femtoseconds,
        comFrequency=com_frequency,
    )
    from .runtime import select_platform
    cuda = select_platform("CUDA")
    logging.info("Production acceleration: cuPotential on CUDA (COM removal every %s steps).", com_frequency)
    return converted, integrator, cuda


def _version(distribution):
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def production_backend(args) -> dict:
    """Reject backend changes before touching a checkpoint or creating a Context."""
    acceleration = getattr(args, "acceleration", "openmm")
    record = {"schema": 1, "acceleration": acceleration, "openmm": _version("openmm")}
    if acceleration == "cupotential":
        record["nv_openmm_plugin"] = _version("nv-openmm-plugin")
    directory = Path(args.output_dir)
    path = directory / "production_backend.json"
    if path.exists():
        try:
            previous = json.loads(path.read_text())
            if not isinstance(previous, dict) or previous.get("schema") != 1:
                raise ValueError("unsupported or missing schema")
            if previous.get("acceleration") != acceleration:
                raise AccelerationError("Production backend differs from this run's recorded backend. Use a new output directory.")
            if args.restart and acceleration == "cupotential" and previous != record:
                raise AccelerationError("cuPotential restart requires the recorded OpenMM and plugin versions.")
        except (OSError, ValueError) as exc:
            raise AccelerationError(f"Cannot read production backend record {path}: {exc}") from exc
    elif acceleration == "cupotential":
        existing = any(directory.glob("production*.chk")) or any(directory.glob("production*.dcd")) or any(directory.glob("production*.csv"))
        if args.restart or existing:
            raise AccelerationError(
                "No cuPotential backend record exists for these production artifacts. "
                "Stock OpenMM checkpoints cannot be resumed with cuPotential; use a new output directory."
            )
    return record


def write_production_backend(args, record: dict) -> None:
    path = Path(args.output_dir) / "production_backend.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def check_cupotential() -> str:
    """Exercise a real CUDA Context, force serialization and the plugin integrator."""
    load_cupotential()
    system = mm.System()
    system.addParticle(39.9)
    system.addParticle(39.9)
    system.setDefaultPeriodicBoxVectors(mm.Vec3(3, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3))
    force = mm.NonbondedForce()
    force.setNonbondedMethod(force.PME)
    force.setCutoffDistance(1 * unit.nanometer)
    force.addParticle(0.1, 0.3, 0.1)
    force.addParticle(-0.1, 0.3, 0.1)
    system.addForce(force)
    system.addForce(mm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin, 0))
    converted, integrator, cuda = cupotential_dynamics(system, 2.0)
    restored = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(converted))
    context = mm.Context(restored, integrator, cuda)
    context.setPositions([mm.Vec3(1, 1, 1), mm.Vec3(1.5, 1, 1)])
    context.setVelocitiesToTemperature(300 * unit.kelvin)
    integrator.step(1)
    energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    if not math.isfinite(energy):
        raise AccelerationError("cuPotential CUDA probe returned nonfinite energy.")
    return f"operational (nv-openmm-plugin {_version('nv-openmm-plugin')}, CUDA Context, XML roundtrip, finite energy)"
