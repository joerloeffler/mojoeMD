<p align="center">
  <img src="mojoeMD_logo.png" alt="mojoeMD_logo" width="420">
</p>

# mojoeMD

`mojoeMD` is a JSON-driven molecular dynamics workflow for OpenMM. It prepares solvated systems, equilibrates them, runs production MD, and includes companion tools for trajectory analysis, contact analysis, interaction-energy screening, and noncanonical amino acid (NCAA) residue-template generation.

The project is aimed at protein systems that may include:

- Standard proteins
- Protein-ligand complexes
- Glycoproteins with GLYCAM templates
- Embedded noncanonical amino acids supplied as OpenMM `ffxml`
- Combinations of glycoproteins, ligands, and NCAAs

## Repository Layout

```text
mojoeMD.py                                      Main OpenMM prepare/equilibrate/production pipeline
analysis/analyse_MD.py                         Trajectory post-processing and structural analyses
analysis/contact_analysis.py                   getcontacts wrapper, contact mapping, optional energies/plots
ncaa/generate_ncaa_ffxml_from_amber_mol2.py    NCAA ffxml/hydrogen-definition generator
examples/*.json                                Example simulation configs
analysis/*.json                                Example analysis/contact-analysis configs
examples/ncaa/                                 Example generated NCAA files
examples/ncaa_work/                            Example NCAA intermediate files
```

## What The Main Pipeline Does

`mojoeMD.py` has three stages:

1. `prepare`
   - Loads a receptor PDB with PDBFixer.
   - Replaces ordinary nonstandard residues unless NCAA mode preserves them.
   - Adds hydrogens with OpenMM `Modeller`.
   - Applies metal-aware histidine/cysteine protonation when retained metals are nearby.
   - Optionally adds a ligand from SDF using OpenFF/RDKit and either Espaloma or SMIRNOFF.
   - Optionally detects and renames glycan residues for GLYCAM.
   - Optionally patches NCAA internal and peptide bonds from supplied `ffxml`.
   - Adds solvent and ions.
   - Writes `prepared.pdb` and `system.xml`.

2. `equilibrate`
   - Loads `prepared.pdb` and `system.xml`.
   - Minimizes energy.
   - Heats from 5 K to 300 K under NVT.
   - Runs NPT equilibration with decreasing backbone restraints.
   - Writes `equilibrated.pdb`, `equilibrated.xml`, and `equilibrated_system.xml`.

3. `produce`
   - Loads the equilibrated system/state.
   - Runs production MD.
   - Writes `production.dcd`, `production.csv`, `production.chk`, and `production_final.chk`.
   - Supports restart from the newest production checkpoint with `--restart`.

## Installation

Create an environment with the scientific MD stack used by the scripts. Exact package names vary by platform, but a typical Conda setup is:

```bash
conda create -n mojoemd -c conda-forge python=3.11 openmm pdbfixer mdtraj numpy scipy scikit-learn tqdm pandas matplotlib
conda activate mojoemd
```

For ligand support, also install OpenFF/RDKit/OpenMM force-field generators:

```bash
conda install -c conda-forge openff-toolkit rdkit openmmforcefields
```

For contact analysis, install `getcontacts` separately and either put its commands on `PATH` or set `getcontacts.path` in `analysis/contact.json`.

## Quick Start

Run a complete standard protein workflow:

```bash
python mojoeMD.py -c examples/01_vanilla_protein.json
```

Run only preparation:

```bash
python mojoeMD.py -c examples/01_vanilla_protein.json --prepare
```

Run production from an existing equilibrated state:

```bash
python mojoeMD.py -c examples/01_vanilla_protein.json --produce
```

Restart production from the latest checkpoint:

```bash
python mojoeMD.py -c examples/01_vanilla_protein.json --produce --restart
```

Configuration values from JSON become command-line defaults. Any explicit CLI argument overrides the config value.

## Example Simulation Config

```json
{
  "receptor": "inputs/protein.pdb",
  "output_dir": "runs/vanilla_protein",
  "prepare": true,
  "equilibrate": true,
  "produce": true,
  "ph": 7.4,
  "water_model": "tip3p",
  "timestep_fs": 4.0,
  "hydrogen_mass": 1.5,
  "prod_ns": 100.0
}
```

Common fields:

- `receptor`: input PDB.
- `ligand`: optional ligand SDF.
- `ligand_resname`: optional 3-character ligand residue name.
- `output_dir`: run directory.
- `prepare`, `equilibrate`, `produce`: stage toggles.
- `prod_ns`: production length in nanoseconds.
- `timestep_fs`: integrator timestep.
- `hydrogen_mass`: hydrogen mass for H-mass repartitioning.
- `ph`: pH used during hydrogen addition.
- `water_model`: solvent model passed to `Modeller.addSolvent`.
- `metal_aware_protonation`: enabled by default; forces metal-aware residue variants such as histidine tautomers and deprotonated cysteine when donor atoms are close to a retained metal.
- `metal_coordination_restraints`: optional harmonic restraints between retained metals and nearby N/O/S donor atoms.
- `metal_coordination_cutoff`: metal-donor detection cutoff in Angstrom.
- `metal_coordination_k`: restraint force constant in kJ/mol/nm^2.

## Protein-Ligand Systems

Use a config such as `examples/02_protein_ligand.json`:

```json
{
  "receptor": "inputs/protein.pdb",
  "ligand": "inputs/ligand.sdf",
  "ligand_resname": "LIG",
  "output_dir": "runs/protein_ligand",
  "prepare": true,
  "equilibrate": true,
  "produce": true,
  "lig_ff": "espaloma"
}
```

`lig_ff` can be:

- `espaloma`
- `smirnoff`

The ligand SDF must contain at least one conformer.

For metalloprotein or metal-coordinated ligand systems, use a config such as
`examples/10_metal_coordinated_protein_ligand.json`. Metal-aware protonation is
enabled by default, and that example also enables harmonic metal coordination
restraints.

## Glycoproteins

Set `glycoprotein: true` and provide GLYCAM resources:

```json
{
  "glycoprotein": true,
  "glycam_xml": "amber14/GLYCAM_06j-1.xml",
  "glycam_hydrogens": "glycam-hydrogens.xml"
}
```

The preparation code detects supported glycan links from PDB `LINK` records and/or distance heuristics, assigns GLYCAM templates, renames glycosylated ASN residues to `NLN` variants, and writes glycan diagnostics under:

```text
<output_dir>/glycan_prepare/
```

Useful glycoprotein debug flags:

```bash
python mojoeMD.py -c examples/09_prepare_only_debug_glycoprotein.json
```

Debug output includes a pre-hydrogen PDB and residue/template bond dumps.

## Noncanonical Amino Acids

NCAA mode preserves configured nonstandard residue names and requires matching OpenMM templates:

```json
{
  "ncaa": true,
  "ncaa_resnames": ["NCA"],
  "ncaa_ffxml": ["ffxml/NCA.xml"],
  "ncaa_mode": "ffxml"
}
```

During preparation, the pipeline:

- Loads custom NCAA `ffxml` files.
- Loads sibling hydrogen-definition files named like `NCA_hydrogens.xml` when present.
- Patches missing internal NCAA bonds from the residue `Bond` records in the `ffxml`.
- Patches peptide bonds adjacent to NCAA residues.

## Generating NCAA Templates

The helper script can extract a local charge fragment and build an OpenMM residue template from an AmberTools/antechamber MOL2 file.

Extract a fragment first:

```bash
python ncaa/generate_ncaa_ffxml_from_amber_mol2.py \
  --protein-pdb inputs/protein_ncaa.pdb \
  --resname ALM \
  --chain A \
  --resid 42 \
  --extract-fragment \
  --fragment-only \
  --workdir ncaa_work
```

After generating a charged MOL2 externally, build the template:

```bash
python ncaa/generate_ncaa_ffxml_from_amber_mol2.py \
  --protein-pdb inputs/protein_ncaa.pdb \
  --mol2 ncaa_work/ALM_amber.mol2 \
  --resname ALM \
  --chain A \
  --resid 42 \
  --outdir ncaa \
  --force
```

Outputs:

- `ncaa/ALM.xml`
- `ncaa/ALM_hydrogens.xml`
- `ncaa/ALM_charge_report.json`

Note: the script can use approximate ff14SB-compatible atom-type fallbacks, but those are compatibility aids, not a replacement for careful parameterization of novel chemistry.

## Trajectory Analysis

Run:

```bash
python analysis/analyse_MD.py -c analysis/analyse.json
```

The analysis pipeline can:

- Strip water and ions.
- Repair periodic boundary artifacts for multichain complexes.
- Pull small molecules back near a protein anchor selection.
- Center and align the trajectory.
- Compute RMSD.
- Compute RMSF and write B-factor-mapped PDBs.
- Compute global and residue-wise SASA.
- Compute dihedral timeseries and summaries.
- Run PCA.
- Run hierarchical clustering and write representative structures.

Typical outputs include:

```text
analysis_out/rmsd.csv
analysis_out/processed_topology.pdb
analysis_out/processed.dcd
analysis_out/rmsf_atoms.csv
analysis_out/rmsf_residues.csv
analysis_out/bfactor_rmsf.pdb
analysis_out/sasa_global.csv
analysis_out/pca_projection.csv
analysis_out/clusters.csv
analysis_out/cluster_representatives/
```

## Contact Analysis

Run:

```bash
python analysis/contact_analysis.py -c analysis/contact.json
```

`contact_analysis.py` can:

- Run `getcontacts` dynamically or statically.
- Parse existing getcontacts TSV files.
- Filter contacts by type and frequency.
- Exclude `wb2` water-bridge contacts.
- Map contact scores onto PDB B-factors.
- Write PyMOL scripts for contact visualization.
- Optionally compute simple nonbonded interaction-energy estimates from `system.xml` or Amber `prmtop`.
- Optionally write residue energy plots and an interactive flareplot.

Key outputs:

```text
contact_analysis_out/getcontacts_A_B.tsv
contact_analysis_out/contacts_filtered.csv
contact_analysis_out/contacts_mapped_bfactor.pdb
contact_analysis_out/contacts.pml
contact_analysis_out/interaction_energies.csv
contact_analysis_out/energies_mapped_bfactor.pdb
contact_analysis_out/energy_contacts.pml
contact_analysis_out/interaction_summary_residue_energies.png
contact_analysis_out/interaction_summary_flareplot.html
```

The energy calculation is a lightweight Coulomb + Lennard-Jones screen. It is useful for diagnostics and ranking, but it is not a rigorous binding free-energy method.

## Important Notes

- Input PDB quality matters. Missing atoms, ambiguous chain IDs, unsupported glycan linkages, or unmatched NCAA residues will stop preparation.
- Metal sites need special care. Metal-aware protonation prevents hydrogens from being added to coordinating histidine nitrogens or cysteine sulfurs, but monatomic metal ions are still nonbonded particles unless `metal_coordination_restraints` or a dedicated metal-site model is used.
- `mojoeMD.py` currently uses Amber14 protein and TIP3P-FB force-field XML files by default.
- `water_model` is passed to OpenMM solvent building; keep it compatible with the loaded water XML.
- Production reports DCD frames every 50 ps.
- Production checkpointing is configured at a long interval; final checkpoint is always written at the end of a successful run.
- GPU platforms use mixed precision, and CUDA sets deterministic forces by default.

## Development Check

Basic syntax check:

```bash
python -m py_compile mojoeMD.py analysis/analyse_MD.py analysis/contact_analysis.py ncaa/generate_ncaa_ffxml_from_amber_mol2.py
```

Most functional checks require installed MD dependencies and realistic molecular input files.

## License

MIT License

Copyright (c) 2026 mojoeMD authors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
