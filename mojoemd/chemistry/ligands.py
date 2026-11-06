"""Ligand preparation and full Espaloma / SMIRNOFF parameterization."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import logging


_has_openff = (
    importlib.util.find_spec("openff.toolkit") is not None
    and importlib.util.find_spec("rdkit") is not None
    and importlib.util.find_spec("openmmforcefields") is not None
)


def pdb_safe_residue_name(name: str) -> str:
    """Return a PDB-safe 3-character residue name."""
    cleaned = "".join(char for char in name.upper() if char.isalnum())
    if not cleaned:
        return "LIG"
    return cleaned[:3]


def prepare_ligand(lig_sdf_file: str | Path, residue_name: str | None = None):
    if not _has_openff:
        raise ImportError("OpenFF Toolkit and RDKit are required for ligand preparation.")

    from rdkit.Chem import BondType, SDMolSupplier

    logging.info(f"Loading ligand from {lig_sdf_file}...")
    rdkit_mol = SDMolSupplier(str(lig_sdf_file))[0]

    if rdkit_mol is None:
        raise ValueError(f"Could not read ligand from SDF file: {lig_sdf_file}")

    invalid_aromatic_bonds = [
        bond for bond in rdkit_mol.GetBonds()
        if bond.GetBondType() == BondType.AROMATIC and not bond.IsInRing()
    ]
    if invalid_aromatic_bonds:
        raise ValueError(
            f"Invalid ligand bond encoding in {lig_sdf_file}: "
            f"{len(invalid_aromatic_bonds)} aromatic bond(s) outside rings. "
            "Provide explicit single/double bond orders and formal charges; "
            "automatic correction would require choosing the ligand's chemistry."
        )

    from openff.toolkit import Molecule

    ligand = Molecule.from_rdkit(rdkit_mol, allow_undefined_stereo=True)
    ligand.name = pdb_safe_residue_name(residue_name or Path(lig_sdf_file).stem)
    logging.info(f"Using ligand residue name: {ligand.name}")
    return ligand


def parametrize_ligand(ligand, forcefield: ForceField, lig_ff: str = "espaloma") -> tuple[Topology, Quantity]:
    from openff.toolkit import Topology as offTopology
    from openff.units.openmm import to_openmm as offquantity_to_openmm
    from openmmforcefields.generators import EspalomaTemplateGenerator, SMIRNOFFTemplateGenerator

    logging.info(f"Parameterizing ligand with {lig_ff}...")

    if lig_ff == "espaloma":
        available = tuple(EspalomaTemplateGenerator.INSTALLED_FORCEFIELDS)
        if not available:
            raise RuntimeError("No Espaloma force field is installed with openmmforcefields.")
        cache_path = Path.home() / ".espaloma"
        cached_models = sorted(cache_path.glob("espaloma-*.pt"))
        espaloma_forcefield = (
            str(cached_models[-1])
            if cached_models
            else sorted(available)[-1]
        )
        source = "cached" if cached_models else "supported"
        logging.info(f"Using {source} Espaloma force field: {espaloma_forcefield}")
        template_generator = EspalomaTemplateGenerator(
            molecules=ligand,
            forcefield=espaloma_forcefield,
        )
    elif lig_ff == "smirnoff":
        template_generator = SMIRNOFFTemplateGenerator(molecules=ligand, forcefield="openff-2.1.0")
    else:
        raise ValueError("Unsupported ligand force field. Choose 'espaloma' or 'smirnoff'.")

    forcefield.registerTemplateGenerator(template_generator.generator)

    ligand_off_topology = offTopology.from_molecules(molecules=[ligand])
    ligand_omm_topology = ligand_off_topology.to_openmm()
    for residue in ligand_omm_topology.residues():
        residue.name = pdb_safe_residue_name(ligand.name)

    if ligand.conformers is None or len(ligand.conformers) == 0:
        raise ValueError("Ligand must have at least one conformer for parameterization.")

    ligand_positions = offquantity_to_openmm(ligand.conformers[0])

    logging.info("Ligand parameterization complete.")
    return ligand_omm_topology, ligand_positions

