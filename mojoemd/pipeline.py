"""Validation and orchestration of molecular-dynamics stages."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from mojoemd.config import PROTEIN_FORCE_FIELDS, parse_arguments, validate_input_files
from mojoemd.equil import run_equilibrate
from mojoemd.prep import run_prepare
from mojoemd.production import run_production
from mojoemd.runtime import setup_logging


def execute_pipeline(args: argparse.Namespace) -> None:
    """Validate and execute stages represented by an argument namespace."""

    if not args.output_dir:
        print("ERROR: The --output_dir argument is required.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    log_file = os.path.join(args.output_dir, "pipeline.log")
    setup_logging(log_file)

    if args.config:
        logging.info(f"Loaded configuration from {args.config}")

    if args.ncaa and not args.ncaa_resnames:
        logging.error("--ncaa requires --ncaa_resnames, e.g. --ncaa_resnames ALM")
        sys.exit(1)

    if args.ncaa and not args.ncaa_ffxml:
        logging.error("--ncaa currently requires --ncaa_ffxml, e.g. --ncaa_ffxml ncaa/ALM.xml")
        sys.exit(1)

    if not (args.prepare or args.equilibrate or args.produce):
        logging.error("No action requested. Please specify at least one stage: --prepare, --equilibrate, or --produce.")
        sys.exit(1)

    if args.prepare and not args.receptor:
        logging.error("The --receptor argument is required.")
        sys.exit(1)

    if args.prepare:
        validate_input_files(args)

    if args.prepare:
        run_prepare(args)

    if args.equilibrate:
        run_equilibrate(args)

    if args.produce:
        run_production(args)

    logging.info("All defined steps completed.")


def main(argv: list[str] | None = None) -> None:
    """Backward-compatible command-line entry point."""
    execute_pipeline(parse_arguments(argv))



if __name__ == "__main__":
    main()
