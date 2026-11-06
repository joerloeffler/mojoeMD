# Testing

The fast tests are unit tests for configuration and API behavior:

```bash
python -m unittest discover -s tests -v
```

Tests that run PDBFixer, hydrogen addition, parameterization, solvent/membrane
construction and OpenMM system serialization together are **preparation
integration tests**. Repeating them after changes also makes them regression
tests. They are opt-in because they take longer and require the molecular
modeling dependencies from `environment.yml`.

## Preparation suites

```bash
# List cases without running preparation.
python tests/preparation_matrix.py --suite matrix --list

# Smaller selection: all protein models, individual modules, full combinations,
# CHARMM membranes, and expected configuration rejections.
python tests/preparation_matrix.py --suite smoke --jobs 2

# Complete feature matrix described below.
python tests/preparation_matrix.py --suite matrix --jobs 2

# Large local structures, including 9UTB.
python tests/preparation_matrix.py --suite stress

# Matrix and large regressions together.
python tests/preparation_matrix.py --suite all --jobs 2
```

The matrix contains 121 cases with both ligand methods enabled:

| Coverage | Cases |
| --- | ---: |
| ff14SB and ff19SB: all 16 on/off combinations of glycan, membrane, ligand and NCAA modules; ligand cases use both SMIRNOFF and Espaloma | 48 |
| Other nine Amber protein models: plain protein, ligand, membrane and membrane–ligand; ligand cases use both methods | 54 |
| Both CHARMM releases: plain protein and membrane | 4 |
| Expected early rejections: unsupported GLYCAM models and CHARMM ligand/NCAA combinations | 15 |

NCAA fixtures use the existing Amber SB-compatible AIB parameters. They do not
assert that those parameters can be reused with IPQ or legacy Amber. Module
combinations are tested with a small, real covalently linked glycan fixture;
the fragment coordinates are a software test setup, not a biological membrane
model. See [fixture provenance](fixtures/preparation/README.md).

The six stress cases run each of these with both ligand methods:

- 9UTB protein plus its bound ligand in a POPC membrane, using local inputs in
  `dev/membrane/`. The supplied 9UTB files contain **no modeled glycans**, so
  this is a membrane–ligand test, not a glycoprotein test.
- The full local glycan/AIB structure plus test ligand and membrane, using
  `dev/refactor_retests/ncaa_glycan/`. This exercises all four modules with a
  larger glycan tree. It is a software stress case, not a biological membrane model.
- The local copper-site protein in `dev/inputs/` plus the known-valid test
  ligand, with metal-aware protonation and metal coordination restraints enabled.
  The original `dev/inputs/ligand.sdf` has six non-ring bonds encoded as aromatic;
  it is retained as a negative unit-test fixture. Preparation rejects this
  encoding rather than silently guessing bond orders or protonation states.

These large inputs are ignored development assets, so a fresh checkout needs
them supplied at those paths. Missing inputs are failures, not silent skips.
The committed small matrix fixtures need no structural downloads. Espaloma
requires its model; the engine uses an existing `~/.espaloma/espaloma-*.pt`
cache when available. Some older OpenMM installations may lack newer force-field
XML files; those cases fail explicitly rather than disappearing from coverage.

## Checks and artifacts

Each case runs in a separate process, so OpenMM hydrogen definitions and
registered ligand templates cannot leak between cases. Every run gets a new
directory under `dev/preparation_tests/`; existing outputs are never overwritten.
Each successful preparation retains its input config, console/pipeline logs,
prepared PDB, serialized system, glycan diagnostics when applicable, and a JSON
result. Expected-rejection cases retain their config, error log and JSON result.
`manifest.json` records software versions and fixture hashes; `summary.json`
records every pass/failure. A failing case gives the runner a nonzero exit code.

Checks include:

- Successful preparation, deserializable system and matching atom/particle counts.
- Finite coordinates, consistent periodic boxes and a box large enough for PME.
- Solvent and membrane lipids present, ligand atom composition retained.
- NCAA atoms/hydrogens and peptide bonds retained.
- Glycan template assignments and all detected covalent glycan links retained.
- Copper atoms retained and metal restraints involving the copper site present.
- Finite initial potential energy and forces, evaluated without integrating.
- Equilibration/production methods forbidden and no trajectories/checkpoints produced.

Only preparation runs. OpenMM's `addHydrogens()` and `addMembrane()` perform
their own minimization/packing relaxation as part of preparation. The tests do
not launch the pipeline's equilibration or production stages. Finite initial
energy does not establish that a prepared structure is equilibrated or suitable
for scientific production.

To select a subset or retain results in a chosen new directory:

```bash
python tests/preparation_matrix.py --suite matrix --ligand-ff smirnoff --match glycoprotein
python tests/preparation_matrix.py --suite stress --match 9utb --output dev/preparation_tests/my-9utb-run
```

`--timeout` sets the per-case limit in seconds (default 1800); `--jobs` limits
concurrent preparation processes. Keep concurrency low for large membrane cases.

The same runner is available through opt-in unittest discovery:

```bash
MOJOEMD_PREPARE_INTEGRATION=1 MOJOEMD_PREPARE_SUITE=matrix MOJOEMD_PREPARE_JOBS=2 \
  python -m unittest discover -s tests -p 'test_prepare_integration.py' -v
```

The suite covers the listed feature combinations and parameter sets. It does
not enumerate every glycan chemistry, NCAA parameter set, water model or pH.

The [2026-10-01 validation record](validation_2026-10-01.md) documents the first
complete run, the issues it exposed and the targeted follow-up checks.


## Modular engine and complete workflow checks

Set `MOJOEMD_TEST_BACKEND=modular` to run the preparation runner against the
independent extracted engine. The runner now defaults to `modular`; set
`MOJOEMD_TEST_BACKEND=legacy` to check the original implementation.
`test_modular.py` checks source/configuration parity and backend independence.
`workflow_regression.py` runs real preparation, the complete existing
equilibration schedule, production, checkpoint continuation, and analysis.
See [the migration record](../REFACTOR.md) for commands and limits.


## Numbered example preparation checks

`python tests/example_preparation.py --output dev/example-check-new --jobs 2`
runs all 11 numbered example templates with recorded input substitutions, using
the large local structures in `dev/`. Every case forces preparation only and
guards against equilibration/production. The output directory must be new.
Original examples and structures are unchanged. The membrane case uses its
matching `dev/membrane/ligand.sdf`; synthetic ethanol is used for other positive
ligand profiles where no matching valid local ligand is available. The malformed
`dev/inputs/ligand.sdf` is separately required to remain rejected.

See [the 2026-11-06 record](validation_2026-11-06_examples.md) and the retained
`dev/modular_prep_check/manifest.json` for the exact mappings and limits.
