"""Configuration parsing, path resolution, and force-field validation."""

from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import json
import logging
import sys


def resolve_config_path(
    value: str | None,
    config_dir: Path | None,
    *,
    require_existing: bool = True,
) -> str | None:
    """Resolve a filesystem path relative to its JSON configuration."""
    if value is None or config_dir is None:
        return value

    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return str(path)

    config_relative = config_dir / path
    if config_relative.exists() or not require_existing:
        return str(config_relative)

    return value


def resolve_config_paths(args: argparse.Namespace, config_dir: Path | None) -> None:
    if config_dir is None:
        return

    args.receptor = resolve_config_path(args.receptor, config_dir, require_existing=False)
    args.ligand = resolve_config_path(args.ligand, config_dir, require_existing=False)

    if args.ncaa_ffxml:
        args.ncaa_ffxml = [
            resolve_config_path(ffxml, config_dir, require_existing=False)
            for ffxml in args.ncaa_ffxml
        ]

    args.glycam_hydrogens = resolve_config_path(args.glycam_hydrogens, config_dir)
    args.water_ff = resolve_config_path(args.water_ff, config_dir)
    args.lipid_ff = resolve_config_path(args.lipid_ff, config_dir)

    if args.output_dir:
        output_path = Path(args.output_dir).expanduser()
        if not output_path.is_absolute():
            args.output_dir = str((config_dir / output_path).resolve())


def validate_existing_file(path: str | None, label: str) -> bool:
    if path is None:
        return True

    if Path(path).expanduser().exists():
        return True

    logging.error(f"{label} file does not exist: {path}")
    return False


def validate_openmm_app_data_file(path: str | None, label: str) -> bool:
    if path is None:
        return True

    if Path(path).expanduser().exists():
        return True

    app_spec = importlib.util.find_spec("openmm.app")
    if app_spec and app_spec.origin:
        data_path = Path(app_spec.origin).parent / "data" / path
        if data_path.exists():
            return True

    logging.error(f"{label} file does not exist: {path}")
    return False


PROTEIN_FORCE_FIELDS = (
    "amber14/protein.ff14SB.xml",
    "amber14/protein.ff15ipq.xml",
    "amber19/protein.ff19SB.xml",
    "amber19/protein.ff19ipq.xml",
    "amber96.xml", "amber99sb.xml", "amber99sbildn.xml",
    "amber99sbnmr.xml", "amber03.xml", "amber10.xml", "amberfb15.xml",
    "charmm36.xml", "charmm36_2024.xml",
)


def configure_protein_force_field(args: argparse.Namespace) -> None:
    """Choose matching defaults and reject unsupported module combinations."""
    if args.protein_ff not in PROTEIN_FORCE_FIELDS:
        raise ValueError(f"Unsupported protein_ff: {args.protein_ff}")
    charmm = args.protein_ff.startswith("charmm")
    legacy_amber = not charmm and "/" not in args.protein_ff
    family = args.protein_ff.removesuffix(".xml") if charmm else "amber14"
    if args.glycoprotein and args.protein_ff not in (
        "amber14/protein.ff14SB.xml", "amber19/protein.ff19SB.xml",
    ):
        raise ValueError(
            "The GLYCAM glycosylation module requires amber14/protein.ff14SB.xml or "
            "amber19/protein.ff19SB.xml. IPQ, CHARMM, and legacy Amber glycoproteins "
            "are not supported by the current GLYCAM templates."
        )
    if charmm and (args.ligand or args.ncaa):
        raise ValueError(
            "CHARMM protein force fields are not supported with the current ligand/NCAA "
            "modules, which use Amber-compatible parameters and 1-4 scaling."
        )
    if args.water_ff is None:
        args.water_ff = (f"{family}/water.xml" if charmm else
                         "tip3p.xml" if legacy_amber else "amber14/tip3pfb.xml")
    if charmm and not args.water_ff.startswith(f"{family}/"):
        raise ValueError(f"{args.protein_ff} requires a matching {family}/ water force field.")
    if not charmm and args.water_ff.startswith("charmm"):
        raise ValueError("Amber protein force fields cannot use CHARMM water parameters.")
    if legacy_amber and args.water_ff.startswith(("amber14/", "amber19/")):
        raise ValueError(
            "Legacy Amber XML files already contain ion templates. Use a standalone water XML "
            "such as tip3p.xml to avoid duplicate ions from Amber14/Amber19 water files."
        )
    if args.lipid_ff is None:
        args.lipid_ff = args.protein_ff if charmm else "amber14/lipid17.xml"
    if charmm and args.membrane and args.lipid_ff != args.protein_ff:
        raise ValueError("CHARMM membrane parameters are already bundled in protein_ff; omit lipid_ff.")
    if not charmm and args.membrane and args.lipid_ff.startswith("charmm"):
        raise ValueError("Amber protein force fields cannot use CHARMM lipid parameters.")


def validate_input_files(args: argparse.Namespace) -> None:
    ok = validate_existing_file(args.receptor, "Receptor")
    ok = validate_openmm_app_data_file(args.protein_ff, "Protein force field") and ok

    if args.ligand:
        ok = validate_existing_file(args.ligand, "Ligand") and ok

    if args.ncaa_ffxml:
        for ffxml in args.ncaa_ffxml:
            ok = validate_existing_file(ffxml, "NCAA ffxml") and ok

    if args.glycoprotein:
        ok = validate_openmm_app_data_file(args.glycam_xml, "GLYCAM force field") and ok
        ok = validate_openmm_app_data_file(args.glycam_hydrogens, "GLYCAM hydrogen definitions") and ok

    ok = validate_openmm_app_data_file(args.water_ff, "Water force field") and ok

    if args.membrane:
        ok = validate_openmm_app_data_file(args.lipid_ff, "Lipid force field") and ok

    if not ok:
        sys.exit(1)


def parse_arguments(
    argv: list[str] | None = None,
    stage_override: str | None = None,
) -> argparse.Namespace:
    """Parse CLI/config arguments without executing a pipeline stage."""
    conf_parser = argparse.ArgumentParser(
        description="A unified pipeline for OpenMM simulations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    conf_parser.add_argument("-c", "--config", help="Path to JSON config file.", metavar="FILE")
    args, remaining_argv = conf_parser.parse_known_args(argv)

    defaults = {}
    config_dir = None

    if args.config:
        try:
            config_path = Path(args.config).expanduser()
            config_dir = config_path.resolve().parent

            with open(config_path) as f:
                defaults = json.load(f)
                print(f"Loaded configuration from {args.config}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Could not load or parse config file: {e}")
            sys.exit(1)

    parser = argparse.ArgumentParser(
        parents=[conf_parser],
        description="A unified pipeline for OpenMM simulations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-r", "--receptor", help="PDB file for the protein receptor.")
    parser.add_argument("-l", "--ligand", help="SDF file for the ligand optional.")
    parser.add_argument("-o", "--output_dir", help="Directory to save all output files.")

    parser.add_argument("--prepare", action="store_true", help="Run the system preparation stage.")
    parser.add_argument("--equilibrate", action="store_true", help="Run the equilibration stage.")
    parser.add_argument("--produce", action="store_true", help="Run the production MD stage.")

    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart production from the latest available checkpoint in the output directory.",
    )

    parser.add_argument("--prod_ns", type=float, default=1.0, help="Length of production run in nanoseconds.")

    parser.add_argument(
        "--checkpoint_interval_ps",
        type=float,
        default=1000.0,
        help="Interval between rolling production checkpoints in picoseconds.",
    )

    parser.add_argument(
        "--lig_ff",
        type=str,
        choices=["espaloma", "smirnoff"],
        default="espaloma",
        help="Force field for ligand parameterization.",
    )
    parser.add_argument(
        "--ligand_resname",
        type=str,
        default=None,
        help="PDB residue name for the ligand. Defaults to the SDF filename stem, truncated to 3 characters.",
    )

    parser.add_argument("--timestep_fs", type=float, default=4.0, help="Integration timestep in femtoseconds.")

    parser.add_argument(
        "--hydrogen_mass",
        type=float,
        default=1.5,
        help="Mass for hydrogen atoms in amu. Use > 1.0 for H-mass repartitioning.",
    )

    parser.add_argument(
        "--protein_ff",
        choices=PROTEIN_FORCE_FIELDS,
        default="amber14/protein.ff14SB.xml",
        help="OpenMM protein force field. GLYCAM requires ff14SB or ff19SB; CHARMM supports protein-only and membrane preparation.",
    )

    parser.add_argument(
        "--water_model",
        default="tip3p",
        help="Water box model name passed to Modeller.addSolvent. Keep tip3p when using amber14/tip3pfb.xml.",
    )

    parser.add_argument(
        "--water_ff",
        default=None,
        help="OpenMM water XML. Defaults to Amber14/19 TIP3P-FB, legacy Amber standalone TIP3P, or the selected CHARMM family water.xml.",
    )

    parser.add_argument(
        "--solvent_padding",
        type=float,
        default=1.0,
        help="Solvent box padding in nanometers for non-membrane systems.",
    )

    parser.add_argument(
        "--ionic_strength",
        type=float,
        default=0.15,
        help="Ionic strength in molar for solvent or membrane preparation.",
    )

    parser.add_argument(
        "--positive_ion",
        default="Na+",
        help="Positive ion type used for neutralization and salt.",
    )

    parser.add_argument(
        "--negative_ion",
        default="Cl-",
        help="Negative ion type used for neutralization and salt.",
    )

    parser.add_argument(
        "--membrane",
        action="store_true",
        help="Build a lipid membrane and water box around the prepared complex.",
    )

    parser.add_argument(
        "--lipid_type",
        default="POPC",
        help="Lipid type passed to Modeller.addMembrane.",
    )

    parser.add_argument(
        "--lipid_ff",
        default=None,
        help="Membrane lipid XML. Defaults to Amber lipid17 or bundled CHARMM parameters.",
    )

    parser.add_argument(
        "--membrane_padding",
        type=float,
        default=1.0,
        help="Minimum membrane/water box padding in nanometers.",
    )

    parser.add_argument(
        "--membrane_center_z",
        type=float,
        default=None,
        help="Membrane center position along z in nanometers. Defaults to the prepared complex z-midpoint.",
    )

    parser.add_argument(
        "--membrane_center_chain_ids",
        nargs="*",
        default=[],
        help="Optional chain IDs used to auto-select membrane center z.",
    )

    parser.add_argument(
        "--membrane_center_residue_range",
        nargs=2,
        type=int,
        default=None,
        metavar=("START", "END"),
        help="Optional residue-id range used to auto-select membrane center z.",
    )

    parser.add_argument(
        "--membrane_auto_thickness",
        type=float,
        default=3.0,
        help="Hydrophobic slab thickness in nanometers for automatic membrane center detection.",
    )

    parser.add_argument("--ph", type=float, default=7.4, help="pH used for hydrogen addition.")

    parser.add_argument(
        "--strip_caps",
        action="store_true",
        help="Remove ACE/NME terminal caps before hydrogen addition and prepare ordinary termini.",
    )

    parser.add_argument(
        "--metal_aware_protonation",
        "--metal-aware-protonation",
        dest="metal_aware_protonation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Force protonation variants from nearby metal contacts before hydrogen addition "
            "(currently histidine tautomers and cysteine thiolates). "
            "Use --no-metal-aware-protonation to disable."
        ),
    )

    parser.add_argument(
        "--metal_coordination_restraints",
        action="store_true",
        help="Add harmonic restraints between retained metal ions and nearby N/O/S donor atoms.",
    )

    parser.add_argument(
        "--metal_coordination_cutoff",
        type=float,
        default=3.0,
        help="Metal-donor detection cutoff in Angstrom.",
    )

    parser.add_argument(
        "--metal_coordination_k",
        type=float,
        default=50000.0,
        help="Metal coordination restraint force constant in kJ/mol/nm^2.",
    )

    parser.add_argument(
        "--ncaa",
        action="store_true",
        help="Enable noncanonical amino acid handling. Selected nonstandard residues are preserved.",
    )

    parser.add_argument(
        "--ncaa_resnames",
        nargs="*",
        default=[],
        help="Residue names to preserve as NCAAs, e.g. ALM PTR SEP.",
    )

    parser.add_argument(
        "--ncaa_ffxml",
        nargs="*",
        default=[],
        help="Custom OpenMM ffxml files containing NCAA residue templates.",
    )

    parser.add_argument(
        "--ncaa_mode",
        choices=["ffxml"],
        default="ffxml",
        help="NCAA parameterization mode. Current robust mode is ffxml.",
    )


    parser.add_argument(
        "--glycoprotein",
        action="store_true",
        help="Enable GLYCAM preparation for covalently glycosylated proteins during --prepare.",
    )

    parser.add_argument(
        "--glycam_xml",
        default="amber14/GLYCAM_06j-1.xml",
        help="OpenMM GLYCAM XML file used when --glycoprotein is enabled.",
    )

    parser.add_argument(
        "--glycam_hydrogens",
        default="glycam-hydrogens.xml",
        help="GLYCAM hydrogen definitions XML used by Modeller when --glycoprotein is enabled.",
    )

    parser.add_argument(
        "--glyco_debug",
        action="store_true",
        help="Write glycoprotein debug files before hydrogen addition.",
    )

    parser.add_argument(
        "--glyco_debug_residue_index",
        type=int,
        default=527,
        help="Topology residue index to dump when --glyco_debug is enabled.",
    )

    parser.add_argument(
        "--glyco_no_link_records",
        action="store_true",
        help="Ignore PDB LINK records during glycan detection and use distance detection only.",
    )

    parser.add_argument(
        "--glyco_no_distance_detection",
        action="store_true",
        help="Disable distance-based glycan bond detection and use LINK records only.",
    )

    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_argv)
    # The lightweight config parser consumes this option before the full parser
    # runs. Preserve it for logging and callers inspecting the namespace.
    args.config = conf_parser.parse_known_args(argv)[0].config
    try:
        configure_protein_force_field(args)
    except ValueError as error:
        parser.error(str(error))
    resolve_config_paths(args, config_dir)

    if stage_override is not None:
        if stage_override not in {"prepare", "equilibrate", "produce"}:
            raise ValueError(f"Unknown pipeline stage: {stage_override}")
        args.prepare = stage_override == "prepare"
        args.equilibrate = stage_override == "equilibrate"
        args.produce = stage_override == "produce"

    return args

