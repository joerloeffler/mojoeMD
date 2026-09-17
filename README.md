<p align="center">
  <img src="mojoeMD_logo.png" alt="mojoeMD_logo" width="420">
</p>

# mojoeMD

`mojoeMD` is an installable, JSON-driven molecular dynamics workflow for OpenMM. It prepares solvated systems, equilibrates them, runs production MD, and includes companion tools for trajectory analysis, contact analysis, interaction-energy screening, and noncanonical amino acid (NCAA) residue-template generation. The workflow can be used through the `mojoemd` command or imported as a Python library.

The project is aimed at protein systems that may include:

- Standard proteins
- Protein-ligand complexes
- Membrane protein-ligand complexes
- Glycoproteins with GLYCAM templates
- Embedded noncanonical amino acids supplied as OpenMM `ffxml`
- Combinations of glycoproteins, ligands, and NCAAs

## Repository Layout

```text
mojoemd/                                        Installed command-line and Python interfaces
mojoeMD.py                                      Established OpenMM simulation engine and legacy CLI
pyproject.toml                                  Python package metadata and console entry point
environment.yml                                Reproducible Conda environment
analysis/analyse_MD.py                         Trajectory post-processing and structural analyses
analysis/contact_analysis.py                   getcontacts wrapper, contact mapping, optional energies/plots
ncaa/generate_ncaa_ffxml_from_amber_mol2.py    NCAA ffxml/hydrogen-definition generator
examples/*.json                                Example simulation configs
analysis/*.json                                Example analysis/contact-analysis configs
examples/ncaa/                                 Example generated NCAA files
examples/ncaa_work/                            Example NCAA intermediate files
```

## What The Main Pipeline Does

The main pipeline has three stages:

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
   - Writes numbered trajectory/report segments such as `production_000.dcd` and `production_000.csv`.
   - Keeps a rolling `production.chk` and a numbered final checkpoint for every segment.
   - Supports restart from the newest production checkpoint with `--restart` without overwriting earlier trajectories.

## Installation

From the repository root, create the complete environment and install mojoeMD in editable mode:

```bash
mamba env create -f environment.yml
conda activate mojoeMD
```

The environment includes the OpenMM stack, AmberTools, ligand dependencies,
Espaloma 0.4.0, analysis dependencies, Packmol 21.2.3, and the local package. Confirm the
installation with:

```bash
mojoemd --help
python -c "import mojoemd; print(mojoemd.__version__)"
packmol < /dev/null 2>&1 | head -n 5
```

To update an existing environment after changing `environment.yml`:

```bash
mamba env update -n mojoeMD -f environment.yml --prune
conda activate mojoeMD
python -m pip install --no-deps -e .
```

Because this is an editable installation, ordinary Python source changes are
available immediately. Reinstall only when `pyproject.toml`, package entry
points, or dependencies change:

```bash
conda activate mojoeMD
cd /path/to/mojoeMD
python -m pip install --no-deps --editable .
hash -r
mojoemd doctor
```

If dependencies in `environment.yml` changed, update the environment first:

```bash
mamba env update -n mojoeMD -f environment.yml --prune
conda activate mojoeMD
python -m pip install --no-deps --editable .
```

For contact analysis, install `getcontacts` separately and either put its commands on `PATH` or set `getcontacts.path` in `analysis/contact.json`.

## Quick Start

For a standard soluble protein preparation, no JSON file is required:

```bash
mojoemd prepare --pdb TEST.pdb
```

This uses the documented defaults and writes beside the input PDB to
`TEST_mojoemd/`. Choose another location with `--output-dir`. Common direct
options include:

```bash
mojoemd prepare --pdb TEST.pdb --output-dir prepared_TEST --ph 7.0
mojoemd prepare --pdb TEST.pdb --ligand LIGAND.sdf
mojoemd prepare --pdb TEST.pdb --membrane
mojoemd run --pdb TEST.pdb
```

`mojoemd run --pdb TEST.pdb` runs preparation, equilibration, and production
with standard settings. JSON remains preferable for production work because it
records every non-default choice, and is required for detailed NCAA,
glycoprotein, metal-site, and membrane configurations.

Run the stages enabled in a configuration:

```bash
mojoemd run -c examples/01_vanilla_protein.json
```

Continue later stages without a JSON file by pointing to the existing run
directory:

```bash
mojoemd equilibrate --output-dir TEST_mojoemd
mojoemd produce --output-dir TEST_mojoemd
mojoemd produce --output-dir TEST_mojoemd --restart
```

Check the environment, Python modules, external programs, and available OpenMM
platforms at any time:

```bash
mojoemd doctor
```

Run only preparation:

```bash
mojoemd prepare -c examples/01_vanilla_protein.json
```

Run production from an existing equilibrated state:

```bash
mojoemd produce -c examples/01_vanilla_protein.json
```

Restart production from the latest checkpoint:

```bash
mojoemd produce -c examples/01_vanilla_protein.json --restart
```

One batch job can run the complete configured workflow; the stages do not need
separate `sbatch` submissions:

```bash
#!/bin/bash
#SBATCH --job-name=mojoemd
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mojoeMD
mojoemd run -c /absolute/path/to/system.json
```

Set `prepare`, `equilibrate`, and `produce` to `true` in that JSON to run all
three sequentially. Separate jobs are useful only when the cluster time limit
or queue policy makes stage-level dependencies preferable.

Every production invocation receives the next free number. A first run writes
`production_000.dcd`, `production_000.csv`, and
`production_final_000.chk`; a restart writes the corresponding `_001` files,
and so on. Existing segments are never copied or overwritten. The rolling
`production.chk` is deliberately replaced at each checkpoint, while completed
numbered checkpoints are retained. Restart selection includes both the new
numbered checkpoints and legacy `production_final.chk` files.

The first `--` separates the installed command's arguments from advanced
engine overrides. For example, `mojoemd prepare -c run.json -- --ph 6.5`
overrides the configured pH. JSON values remain the defaults.

Paths in JSON—including `receptor`, `ligand`, `ncaa_ffxml`, and `output_dir`—are
resolved relative to the JSON file, not the shell's working directory. Absolute
paths remain unchanged.

The historical invocation remains available during the refactor:

```bash
python mojoeMD.py -c examples/01_vanilla_protein.json
```

## Python API

The same workflow can be called from another module without constructing a
subprocess:

```python
from mojoemd import prepare_system, run_pipeline

prepared = prepare_system("configs/system.json")
print(prepared.prepared_pdb, prepared.system_xml)

# Alternatively, honor prepare/equilibrate/produce flags in the JSON file.
result = run_pipeline("configs/system.json")
print(result.output_dir)
```

Stage-specific functions are `prepare_system()`, `equilibrate_system()`, and
`run_production()`. Each returns a `PipelineResult` containing the output
directory and available primary files. Validation failures raise
`PipelineExecutionError` instead of terminating the host Python process.

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
- `checkpoint_interval_ps`: rolling checkpoint interval in picoseconds; defaults to 1000 ps.
- `timestep_fs`: integrator timestep.
- `hydrogen_mass`: hydrogen mass for H-mass repartitioning.
- `ph`: pH used during hydrogen addition.
- `water_model`: solvent model passed to `Modeller.addSolvent`.
- `water_ff`: OpenMM water force-field XML file.
- `membrane`: build a lipid membrane and water box with `Modeller.addMembrane`.
- `lipid_type`: lipid type for OpenMM membrane building, for example `POPC`.
- `lipid_ff`: OpenMM lipid force-field XML file used for membrane systems.
- `membrane_padding`: minimum membrane/water padding in nanometers.
- `membrane_center_z`: explicit membrane center along z in nanometers.
- `membrane_center_chain_ids`: optional chain IDs used to auto-center the membrane.
- `membrane_center_residue_range`: optional residue-id range used to auto-center the membrane.
- `membrane_auto_thickness`: hydrophobic slab thickness used for automatic TM-window detection.
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

## Membrane Systems

Use a config such as `examples/11_membrane_ligand.json`:

```json
{
  "receptor": "inputs/membrane_protein.pdb",
  "ligand": "inputs/ligand.sdf",
  "output_dir": "runs/membrane_ligand",
  "prepare": true,
  "membrane": true,
  "lipid_type": "POPC",
  "lipid_ff": "amber14/lipid17.xml",
  "water_ff": "amber14/tip3p.xml",
  "membrane_center_z": null,
  "membrane_center_chain_ids": [],
  "membrane_center_residue_range": null,
  "membrane_auto_thickness": 3.0
}
```

Membrane mode prepares the protein and optional ligand first, then calls OpenMM
`Modeller.addMembrane()` to build the lipid bilayer, water, and ions around the
complex. The input structure should already be oriented for a membrane in the XY
plane.

If `membrane_center_z` is set, that explicit z value is used. If it is `null`
and no chain/range selection is provided, mojoeMD scans the oriented structure
for a hydrophobic, z-spanning sequence window and centers the bilayer on that
predicted transmembrane slab. `membrane_auto_thickness` controls the slab window
in nanometers. If the heuristic is wrong for a difficult structure, set
`membrane_center_chain_ids` and `membrane_center_residue_range` to anchor the
center on known transmembrane residues.

OpenMM's built-in membrane builder creates a single-lipid membrane from the
selected `lipid_type`. For mixed or asymmetric membrane compositions, build the
membrane externally and then use mojoeMD with force-field files that support the
lipid residue names in that prepared structure.

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
mojoemd prepare -c examples/09_prepare_only_debug_glycoprotein.json
```

Debug output includes a pre-hydrogen PDB and residue/template bond dumps.

## Noncanonical Amino Acids

NCAA mode preserves explicitly selected nonstandard residues and loads matching
OpenMM residue templates. Identify each target by residue name, chain ID,
residue number, and insertion code where applicable. Do not assume every
`HETATM` residue is an NCAA: crystallographic ligands, terminal modifications,
metals, and known modified amino acids require separate decisions.

Before parameterization, inspect all nonstandard residues and determine whether
each one should be preserved, replaced, removed, or parameterized. For example,
Amber14 already contains force-field templates for hydroxyproline (`HYP`), but
OpenMM's default hydrogen definitions may still be insufficient to add its
hydrogens. Conversely, an unsupported terminal residue such as pyroglutamate
(`PCA`) requires a terminally appropriate custom template if it is retained.

A preparation configuration looks like:

```json
{
  "ncaa": true,
  "ncaa_resnames": ["NCA"],
  "ncaa_ffxml": ["ffxml/NCA.xml"],
  "ncaa_mode": "ffxml"
}
```

During preparation, the pipeline:

- Tells PDBFixer which nonstandard residue names must remain unchanged.
- Loads custom NCAA `ffxml` residue templates.
- Loads sibling hydrogen-definition files named like `NCA_hydrogens.xml` when present.
- Patches missing internal NCAA bonds from the residue `Bond` records in the `ffxml`.
- Patches peptide bonds adjacent to NCAA residues.
- Adds hydrogens with OpenMM `Modeller` and requires every residue to match a
  loaded force-field template before solvation.

Names listed in `ncaa_resnames` are preserved. If a known modified residue must
remain present, list it too—even when its force-field template is already part
of Amber—or PDBFixer may replace it. Only custom XML files belong in
`ncaa_ffxml`.

## Generating NCAA Templates

The helper extracts a local charge fragment and converts an AmberTools MOL2
containing AM1-BCC charges into an OpenMM residue template. The extracted
fragment normally includes the NCAA plus neighboring residues so that its
charges reflect the peptide environment.

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

Before running Antechamber, ensure the fragment contains its complete and
chemically correct hydrogen set. Antechamber assigns charges and atom types; it
does not reliably add missing hydrogens. A heavy-atom-only fragment can produce
the wrong net charge, an odd-electron calculation, or an incomplete OpenMM
template.

Run Antechamber on the complete fragment, explicitly setting the charge of the
**whole extracted fragment**, not merely the target residue. One representative
command is:

```bash
antechamber \
  -i ncaa_work/ALM_fragment_complete_h.pdb -fi pdb \
  -o ncaa_work/ALM_fragment_amber.mol2 -fo mol2 \
  -c bcc -nc 0 -at gaff2 -j 5 -dr no -pf y -s 2
```

Check the electron count and SQM convergence rather than changing `-nc` merely
to silence an error. If altered SQM controls are scientifically justified, put
them explicitly in the command and retain the command and `sqm.out` with the
parameterization record.

Then build the template:

```bash
python ncaa/generate_ncaa_ffxml_from_amber_mol2.py \
  --protein-pdb inputs/protein_ncaa.pdb \
  --mol2 ncaa_work/ALM_amber.mol2 \
  --resname ALM \
  --chain A \
  --resid 42 \
  --outdir ncaa \
  --target-charge 0 \
  --charge-correction uniform \
  --type-source protein \
  --atom-type-map ncaa_work/ALM_atom_type_map.json \
  --force
```

Outputs:

- `ncaa/ALM.xml`
- `ncaa/ALM_hydrogens.xml`
- `ncaa/ALM_charge_report.json`

### Charges and atom types

AM1-BCC supplies partial charges, but it does not supply the bonded and
Lennard-Jones parameters needed by OpenMM. The generator therefore needs atom
types already defined by a loaded force field:

- Prefer an explicit `--atom-type-map` based on the NCAA's chemistry.
- `--type-source protein` can reuse suitable ff14SB types for genuinely
  protein-like functional groups.
- `--allow-approximate-types` enables heuristic GAFF/MOL2-to-ff14SB mappings.
  This is a compatibility aid for testing, not automatic publication-grade
  parameterization.
- `--type-source mol2` emits prefixed types, but those types are useful only
  when a separate force-field XML defines their masses and nonbonded/bonded
  parameters.

Set `--target-charge` to the intended formal charge of the NCAA residue itself.
The generator reports the raw charge extracted from the MOL2, the correction
method, and the final residue-template charge. Neutralization is a deliberate
chemical choice, not a default that is correct for every NCAA.

Charge results can depend on neighboring residues and protonation state. If the
same NCAA appears in substantially different chemical contexts, parameterize
and review each context independently or deliberately adopt one transferable,
validated parameter set.

### Required validation

Do not consider XML generation alone a successful parameterization. Run
`mojoemd prepare -c <config.json>` and verify that:

1. The intended nonstandard residues were preserved and unwanted ones were
   removed or replaced explicitly.
2. The NCAA has every expected heavy atom and hydrogen with unique names.
3. Internal and adjacent peptide bonds are present.
4. Every residue matches exactly one force-field template after hydrogen
   addition.
5. The reported NCAA template charge matches the intended formal charge.
6. `prepared.pdb` and the deserialized `system.xml` contain the same number of
   atoms/particles.
7. The preparation log, source fragment, hydrogenated fragment, MOL2, atom-type
   map, XML files, configuration, commands, tool versions, and checksums are
   retained for reproducibility.

Parameters produced with approximate type mappings are appropriate for
pipeline development and compatibility testing. Novel chemistry should receive
independent charge, bonded-parameter, protonation, and conformational validation
before production or publication-quality simulations.

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
- mojoeMD currently uses Amber14 protein and TIP3P-FB force-field XML files by default.
- `water_model` is passed to OpenMM solvent building; keep it compatible with the loaded water XML.
- Production reports DCD frames every 50 ps.
- Production checkpointing is configured at a long interval; final checkpoint is always written at the end of a successful run.
- GPU platforms use mixed precision, and CUDA sets deterministic forces by default.

## Development Check

Basic syntax check:

```bash
python -m unittest discover -s tests -v
python -m py_compile mojoeMD.py mojoemd/*.py analysis/analyse_MD.py analysis/contact_analysis.py ncaa/generate_ncaa_ffxml_from_amber_mol2.py
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
