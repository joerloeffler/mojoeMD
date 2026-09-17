"""Installation diagnostics for mojoeMD."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


MODULES = {
    "OpenMM": "openmm",
    "PDBFixer": "pdbfixer",
    "NumPy": "numpy",
    "tqdm": "tqdm",
    "OpenFF Toolkit": "openff.toolkit",
    "OpenMMForceFields": "openmmforcefields",
    "OpenMMTools": "openmmtools",
    "Espaloma": "espaloma",
    "RDKit": "rdkit",
    "MDTraj": "mdtraj",
    "pandas": "pandas",
    "SciPy": "scipy",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
}

EXECUTABLES = ("packmol", "antechamber")


def _distribution_version(module: str) -> str:
    distribution = {
        "openff.toolkit": "openff-toolkit",
        "sklearn": "scikit-learn",
    }.get(module, module.split(".")[0])
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "installed"


def run_doctor() -> bool:
    """Print dependency diagnostics and return whether required checks pass."""
    print("mojoeMD doctor")
    print("---------------")
    healthy = True

    try:
        from .api import _load_engine

        engine = _load_engine()
        engine_ok = callable(engine.parse_arguments) and callable(engine.execute_pipeline)
        healthy &= engine_ok
        print(f"[{'OK' if engine_ok else '!!'}] Pipeline engine: {engine.__file__}")
    except Exception as exc:
        healthy = False
        print(f"[!!] Pipeline engine: {exc}")

    for label, module in MODULES.items():
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError):
            found = False
        healthy &= found
        detail = _distribution_version(module) if found else "MISSING"
        print(f"[{'OK' if found else '!!'}] {label}: {detail}")

    for executable in EXECUTABLES:
        environment_executable = Path(sys.prefix) / "bin" / executable
        path = (
            str(environment_executable)
            if environment_executable.exists()
            else shutil.which(executable)
        )
        healthy &= path is not None
        print(f"[{'OK' if path else '!!'}] {executable}: {path or 'MISSING'}")

    try:
        from openmm import Platform

        platforms = [
            Platform.getPlatform(index).getName()
            for index in range(Platform.getNumPlatforms())
        ]
        platform_detail = ", ".join(platforms) or "none"
        healthy &= bool(platforms)
        print(f"[{'OK' if platforms else '!!'}] OpenMM platforms: {platform_detail}")
    except Exception as exc:
        healthy = False
        print(f"[!!] OpenMM platforms: {exc}")

    print("Result: " + ("healthy" if healthy else "problems found"))
    return healthy
