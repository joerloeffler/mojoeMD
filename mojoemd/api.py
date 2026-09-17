"""Callable orchestration API backed by the established mojoeMD engine."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


VALID_STAGES = ("prepare", "equilibrate", "produce")


class PipelineExecutionError(RuntimeError):
    """A legacy engine validation or execution step terminated unsuccessfully."""


def _load_engine() -> Any:
    """Load the legacy engine by path, avoiding macOS case-folding collisions."""
    module_name = "_mojoemd_engine"
    if module_name in sys.modules:
        return sys.modules[module_name]

    engine_path = Path(__file__).resolve().parent.parent / "mojoeMD.py"
    spec = importlib.util.spec_from_file_location(module_name, engine_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load mojoeMD engine from {engine_path}")
    engine = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = engine
    spec.loader.exec_module(engine)
    return engine


@dataclass(frozen=True)
class PipelineResult:
    """Paths produced or expected for a completed pipeline invocation."""

    output_dir: Path
    stages: tuple[str, ...]
    pipeline_log: Path
    prepared_pdb: Path | None = None
    system_xml: Path | None = None
    equilibrated_pdb: Path | None = None
    production_dcd: Path | None = None


def _result(output_dir: str | Path, stages: Iterable[str]) -> PipelineResult:
    directory = Path(output_dir).resolve()

    def existing(name: str) -> Path | None:
        path = directory / name
        return path if path.exists() else None

    production_segments = sorted(directory.glob("production_[0-9][0-9][0-9].dcd"))
    latest_production = production_segments[-1] if production_segments else existing("production.dcd")

    return PipelineResult(
        output_dir=directory,
        stages=tuple(stages),
        pipeline_log=directory / "pipeline.log",
        prepared_pdb=existing("prepared.pdb"),
        system_xml=existing("system.xml"),
        equilibrated_pdb=existing("equilibrated.pdb"),
        production_dcd=latest_production,
    )


def run_pipeline(
    config: str | Path | None = None,
    *,
    stages: Sequence[str] | None = None,
    cli_overrides: Sequence[str] = (),
) -> PipelineResult:
    """Run configured stages and return their output paths.

    ``cli_overrides`` uses the compatibility parser syntax, for example
    ``("--ph", "6.5")``. When ``stages`` is omitted, stage booleans are read
    from the JSON configuration.
    """
    requested = tuple(stages) if stages is not None else None
    if requested is not None:
        unknown = sorted(set(requested) - set(VALID_STAGES))
        if unknown:
            raise ValueError(f"Unknown pipeline stages: {', '.join(unknown)}")
    engine = _load_engine()
    argv = (["--config", str(config)] if config is not None else []) + list(cli_overrides)
    try:
        args = engine.parse_arguments(argv)
        if requested is not None:
            for stage in VALID_STAGES:
                setattr(args, stage, stage in requested)
        active = tuple(stage for stage in VALID_STAGES if getattr(args, stage))
        engine.execute_pipeline(args)
    except SystemExit as exc:
        raise PipelineExecutionError(
            f"mojoeMD pipeline stopped with exit status {exc.code}"
        ) from exc
    return _result(args.output_dir, active)


def prepare_system(config: str | Path, *, cli_overrides: Sequence[str] = ()) -> PipelineResult:
    """Run only system preparation."""
    return run_pipeline(config, stages=("prepare",), cli_overrides=cli_overrides)


def equilibrate_system(config: str | Path, *, cli_overrides: Sequence[str] = ()) -> PipelineResult:
    """Run only equilibration using an existing prepared system."""
    return run_pipeline(config, stages=("equilibrate",), cli_overrides=cli_overrides)


def run_production(config: str | Path, *, cli_overrides: Sequence[str] = ()) -> PipelineResult:
    """Run only production using an existing equilibrated system."""
    return run_pipeline(config, stages=("produce",), cli_overrides=cli_overrides)
