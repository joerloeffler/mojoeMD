"""Installation diagnostics for mojoeMD."""

from __future__ import annotations

import importlib
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
    "PyTorch": "torch",
    "DGL": "dgl",
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
        from .api import DEFAULT_BACKEND, _select_engine

        engine = _select_engine(DEFAULT_BACKEND)
        engine_ok = callable(engine.parse_arguments) and callable(engine.execute_pipeline)
        healthy &= engine_ok
        print(f"[{'OK' if engine_ok else '!!'}] Pipeline engine ({DEFAULT_BACKEND}): {engine.__file__}")
    except Exception as exc:
        healthy = False
        print(f"[!!] Pipeline engine: {exc}")

    for label, module in MODULES.items():
        try:
            importlib.import_module(module)
            imported = True
            detail = _distribution_version(module)
        except Exception as exc:
            imported = False
            detail = f"BROKEN ({type(exc).__name__}: {exc})"
        healthy &= imported
        print(f"[{'OK' if imported else '!!'}] {label}: {detail}")

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
        dgl = importlib.import_module("dgl")
        adjacency = dgl.graph(([0], [1]), num_nodes=2).adj()
        sparse_ok = adjacency.shape == (2, 2)
        healthy &= sparse_ok
        print(f"[{'OK' if sparse_ok else '!!'}] DGL/PyTorch sparse backend: operational")
    except Exception as exc:
        healthy = False
        print(f"[!!] DGL/PyTorch sparse backend: {exc}")

    try:
        from openff.toolkit.utils import AmberToolsToolkitWrapper

        ambertools_ok = AmberToolsToolkitWrapper.is_available()
        healthy &= ambertools_ok
        detail = "available" if ambertools_ok else "unavailable; activate the environment so antechamber and sqm are on PATH"
        print(f"[{'OK' if ambertools_ok else '!!'}] OpenFF AmberTools charge backend: {detail}")
    except Exception as exc:
        healthy = False
        print(f"[!!] OpenFF AmberTools charge backend: {exc}")

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
