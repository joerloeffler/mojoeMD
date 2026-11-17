"""Logging, reporters, and OpenMM platform selection."""

from __future__ import annotations

from pathlib import Path
import logging
import sys
import time

from openmm import Platform, State
from openmm.app import Simulation
from openmm.unit import nanoseconds


try:
    from openmmtools.utils import get_fastest_platform
except ImportError:
    def get_fastest_platform() -> Platform:
        preferred = ["CUDA", "OpenCL", "CPU", "Reference"]
        available = {
            Platform.getPlatform(i).getName(): Platform.getPlatform(i)
            for i in range(Platform.getNumPlatforms())
        }
        for name in preferred:
            if name in available:
                return available[name]
        raise RuntimeError("No OpenMM platform is available.")


def setup_logging(log_file: str | Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def select_platform(platform_name: str | None = None) -> Platform:
    if platform_name is None or platform_name.lower() == "fastest":
        platform = get_fastest_platform()
        logging.info(f"Using fastest available platform: {platform.getName()}")
    else:
        platform = Platform.getPlatformByName(platform_name)
        logging.info(f"Using specified platform: {platform.getName()}")

    if platform.getName() in ["CUDA", "OpenCL"]:
        platform.setPropertyDefaultValue("Precision", "mixed")
        if platform.getName() == "CUDA":
            platform.setPropertyDefaultValue("DeterministicForces", "true")

    return platform


class LoggingReporter:
    def __init__(self, report_interval: int = 10000) -> None:
        self._reportInterval = report_interval
        self._last_reported_time = None
        self._last_reported_ns = 0.0

    def describeNextReport(self, simulation: Simulation) -> tuple[int, bool, bool, bool, bool, bool]:
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False, False)

    def report(self, simulation: Simulation, state: State) -> None:
        current_time = time.perf_counter()
        dt = simulation.integrator.getStepSize()
        current_nanoseconds = simulation.currentStep * dt.value_in_unit(nanoseconds)

        if self._last_reported_time is not None:
            elapsed_wall_time = current_time - self._last_reported_time
            elapsed_ns = current_nanoseconds - self._last_reported_ns
            compute_time_s_ns = elapsed_wall_time / elapsed_ns if elapsed_ns > 0 else 0.0
            logging.info(
                f"Step {simulation.currentStep}, ns: {current_nanoseconds:.3f}, "
                f"compute_time: {compute_time_s_ns:.2f} s/ns"
            )
        else:
            logging.info(f"Step {simulation.currentStep}, ns: {current_nanoseconds:.3f}, compute_time: initializing...")

        self._last_reported_time = current_time
        self._last_reported_ns = current_nanoseconds

