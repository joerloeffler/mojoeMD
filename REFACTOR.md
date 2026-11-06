# Modular engine migration

The established implementation in `mojoeMD.py` remains an independent fallback.
Its contents are unchanged by this refactor. Python, PyTorch, DGL, Espaloma,
force-field choices, and simulation protocols remain unchanged.

## Source layout

The existing `mojoemd/` package is the source directory. Keeping its location
preserves the current editable installation and avoids coupling this scientific
refactor to a packaging-layout migration.

| Module | Responsibility |
| --- | --- |
| `api.py`, `cli.py` | Callable API and installed command |
| `config.py` | Configuration, paths, and force-field compatibility |
| `pipeline.py` | Stage validation and orchestration |
| `prep.py` | Preparation orchestration |
| `equil.py` | Minimization, heating, and NPT equilibration |
| `production.py` | Production, checkpoints, and numbered continuation segments |
| `runtime.py` | Logging, reporters, and platform selection |
| `chemistry/protein.py` | Protein repair and template checks |
| `chemistry/ligands.py` | Existing full Espaloma / SMIRNOFF parameterization |
| `chemistry/glycans.py` | GLYCAM chemistry, link detection, and diagnostics |
| `chemistry/ncaa.py` | NCAA bonds and hydrogen definitions |
| `chemistry/metals.py` | Metal-aware protonation and coordination restraints |
| `chemistry/membranes.py` | Membrane centering and periodic-box handling |
| `analyse.py`, `contacts.py` | Existing trajectory and contact analysis |

The standalone `analysis/` scripts are retained independently. The extracted
analysis entry points accept an explicit argument list for library/CLI use.
The CLI imports the analysis modules only when an analysis command is selected.

## Engine selection

As of 2026-11-06, the CLI and Python API default to the modular backend.
`--backend modular` remains valid; `--backend legacy` explicitly selects the
original implementation. Examples:

```bash
mojoemd prepare --pdb protein.pdb -o prepared --backend modular
mojoemd run --config run.json
mojoemd run --config run.json --backend legacy
mojoemd produce --config run.json --backend modular --restart
mojoemd analyse --config analysis.json
mojoemd contacts --config contacts.json
```

```python
from mojoemd import run_pipeline, prepare_system
result = run_pipeline("run.json")
legacy_result = run_pipeline("run.json", backend="legacy")
result = prepare_system("prepare.json", backend="modular")
```

The legacy command remains `python mojoeMD.py --config run.json`. Neither backend
calls into the other. There is no automatic retry through another backend after
failure, which could otherwise duplicate simulation work or hide an error.

## Acceptance gates

1. Capture the baseline, including user changes already present before extraction.
2. Preserve every legacy function/class body and chemistry constant structurally,
   except the explicitly recorded checkpoint-path fix;
   compare parsed configuration for every exposed protein force field.
3. Run the real preparation matrix with the modular backend, retaining artifact
   checks for connectivity, composition, serialization, energy, and forces.
4. Exercise actual full-length equilibration, two short production segments with
   checkpoint continuation, and analysis for both backends in separate processes.
5. Compare analysis CSVs on the same trajectory. Stochastic dynamics and solvent
   packing do not require byte-identical independently prepared structures.
6. Validate all numbered example profiles with explicitly mapped local inputs
   before changing the default backend. This gate passed on 2026-11-06.
   Broader complex-system dynamics remain a validation limitation; preparation
   success alone does not establish equilibration or production suitability.

The source parity test guards this extraction, rather than requiring old and new
implementations to remain identical forever. Future intentional scientific changes
must update that gate deliberately and include targeted scientific validation.

## Reproduce validation

Use the existing validated environment; do not upgrade dependencies for this work.

```bash
python -m unittest discover -s tests -v
python tests/ligand_parity.py
python tests/example_preparation.py --output dev/example-check-new --jobs 2
MOJOEMD_TEST_BACKEND=modular python tests/preparation_matrix.py --suite matrix --jobs 2
python tests/workflow_regression.py --backend legacy --output dev/workflow-legacy
python tests/workflow_regression.py --backend modular --output dev/workflow-modular
```

The workflow runner performs all 130,000 existing equilibration steps and 50 ps
production followed by another 50 ps after restart, with a 2 fs timestep on CPU.
This verifies execution and continuity, not scientific convergence. Each output
directory must be new. The preparation runner supports both backends through
`MOJOEMD_TEST_BACKEND` and records backend selection in its manifest.

Initial local baseline: `dev/refactor_baseline/manifest.json`, the original script,
and `working-tree.patch`. This captures pre-existing changes without making a Git
commit or altering those changes. Large generated validation artifacts stay in
ignored `dev/`; tests and small fixtures are now visible to version control.

## Separate future work

Changing ligand force fields or charge methods, migrating the environment,
adding BioEmu2, and changing membrane equilibration physics are separate changes.

The CLI now retains its parser instance so missing-input errors correctly use
`argparse` instead of raising `NameError`. This error-path fix is covered by
a regression test and does not alter simulation behavior.

## Restart bug found during validation

The original production stage passes a `pathlib.Path` to
`Simulation.loadCheckpoint()`. OpenMM 8.4.0 treats non-string inputs as file
objects, so restart fails with `AttributeError: PosixPath has no attribute read`.
The modular production stage converts the path to `str` before loading. The
legacy backup retains its original contents. The source parity test permits
exactly this call-site conversion; it does not ignore the production function.

## Validation result

The [2026-11-04 validation record](tests/validation_2026-11-04_refactor.md) records
121/121 matrix cases, both copper-site preparations, ligand parameter parity,
the full soluble-system workflow and restart, analysis parity, and wheel checks.
The [2026-11-06 example checks](tests/validation_2026-11-06_examples.md) passed
all 11 resolved profiles, including the full membrane system and matching
ligand. Modular is now the default. Broader complex-system dynamics and external
contact-tool execution remain outside the completed validation.
