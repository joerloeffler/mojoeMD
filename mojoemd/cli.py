"""Installed command-line interface for mojoeMD."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .api import PipelineExecutionError, run_pipeline
from .doctor import run_doctor


BUNNY = "   (\\_/)\n   ( o.o)\n   (> <)  mojoemd"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mojoemd", description="Prepare and simulate molecular systems with OpenMM.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "prepare": "Repair, protonate, parameterize, solvate, and serialize a system.",
        "equilibrate": "Equilibrate an existing prepared system.",
        "produce": "Run production dynamics from an equilibrated system.",
        "run": "Run all stages, or the stages enabled in a JSON configuration.",
    }
    for name, description in descriptions.items():
        command = subparsers.add_parser(name, help=description)
        command.add_argument("-c", "--config", help="Optional JSON configuration file")
        command.add_argument("--pdb", help="Protein PDB; uses standard settings without a JSON file")
        command.add_argument("-o", "--output-dir", help="Output directory")
        command.add_argument("--ligand", help="Optional ligand SDF")
        command.add_argument("--membrane", action="store_true", help="Build the standard POPC membrane system")
        command.add_argument("--ph", type=float, help="Preparation pH")
        if name == "produce":
            command.add_argument(
                "--restart",
                action="store_true",
                help="Continue from the newest production checkpoint.",
            )
        command.add_argument(
            "engine_args",
            nargs=argparse.REMAINDER,
            help="Compatibility overrides passed after '--', for example: -- --ph 6.5",
        )
    subparsers.add_parser("doctor", help="Check the mojoeMD installation and external tools.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    print(BUNNY)
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return 0 if run_doctor() else 1

    if not args.config and args.command in {"prepare", "run"} and not args.pdb:
        parser.error(f"{args.command} requires either --pdb or --config")
    if not args.config and args.command in {"equilibrate", "produce"} and not args.output_dir:
        parser.error(f"{args.command} without --config requires --output-dir")

    stages = None if args.command == "run" and args.config else (
        ("prepare", "equilibrate", "produce") if args.command == "run" else (args.command,)
    )
    overrides = list(args.engine_args)
    if overrides[:1] == ["--"]:
        overrides = overrides[1:]
    if getattr(args, "restart", False):
        overrides.append("--restart")
    if args.pdb:
        overrides.extend(("--receptor", args.pdb))
    output_dir = args.output_dir
    if not output_dir and not args.config and args.pdb:
        pdb_path = Path(args.pdb).expanduser().resolve()
        output_dir = str(pdb_path.with_name(f"{pdb_path.stem}_mojoemd"))
    if output_dir:
        overrides.extend(("--output_dir", output_dir))
    if args.ligand:
        overrides.extend(("--ligand", args.ligand))
    if args.membrane:
        overrides.append("--membrane")
    if args.ph is not None:
        overrides.extend(("--ph", str(args.ph)))

    try:
        result = run_pipeline(args.config, stages=stages, cli_overrides=overrides)
    except PipelineExecutionError as exc:
        parser.error(str(exc))
    print(f"mojoeMD completed {', '.join(result.stages)} -> {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
