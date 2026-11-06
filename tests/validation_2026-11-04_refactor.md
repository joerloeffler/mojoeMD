# Modular-engine validation: 2026-11-04

The modular engine is opt-in through `--backend modular` or
`backend="modular"`. Existing calls still default to the independent legacy
engine. Neither the environment nor ligand force-field choices were changed.

## Completed checks

| Check | Result |
| --- | --- |
| Baseline fast suite | 17 passed, 1 opt-in integration wrapper skipped |
| Updated fast suite | 24 passed, 1 opt-in integration wrapper skipped |
| Legacy source functions/classes and chemistry constants | Preserved structurally, except the recorded checkpoint filename fix |
| Configuration parity | Equal parsed namespaces for every exposed protein force field |
| SMIRNOFF ligand parity | Identical serialized OpenMM systems and coordinates |
| Full Espaloma ligand parity | Identical serialized OpenMM systems and coordinates |
| Legacy ff19SB glycan + membrane + ligand + NCAA comparison | Passed |
| Copper-site preparations, SMIRNOFF and Espaloma | Both passed retained-metal, restraint, and finite-energy/force checks |
| Modular soluble-system workflow | Preparation, full equilibration, production, restart, and analysis passed |
| Analysis parity | RMSD, atom/residue RMSF, and global SASA CSVs byte-identical to the original analysis on the same trajectory |
| Packaging | Wheel builds and contains all modules, chemistry subpackage, and legacy engine; isolated imports pass |

## Preparation matrix

All 121 cases passed: 106 real preparations and 15 expected early configuration
rejections. The matrix covers the exposed protein force fields, glycan/membrane/
ligand/NCAA combinations, and both ligand methods. Together with the two separate
copper-site cases, 108 modular preparations passed retained-chemistry,
serialization, and finite-energy/force checks.

Preparation definitions and chemistry constants match the frozen legacy source.
Import formatting/type-annotation imports were cleaned up during execution;
final source hashes are retained alongside the initial manifest. The production
checkpoint-path correction does not affect this preparation-only matrix.

The large 9UTB membrane-ligand and full 33-sugar stress cases from the October
record were not rerun in this round. Complex-system equilibration/production,
long trajectories, scientific convergence, and external contact-tool execution
remain outside the completed validation. The modular backend therefore remains
opt-in. This is an engineering migration validation, not a scientific validation
of every supported molecular chemistry.

## Workflow details and restart defect

Both backends completed the original 130,000-step equilibration schedule and
an initial 50 ps production segment on CPU at a 2 fs timestep. The modular
backend also completed a 50 ps checkpoint continuation. Reported production
times advanced from approximately 300 ps to 350 ps. Each segment contains one
reported trajectory frame; these short segments test handoffs and restart
continuity, not ensemble quality, long-term stability, or scientific convergence.

The legacy backend's actual restart fails in OpenMM 8.4.0 because
`Simulation.loadCheckpoint()` receives a `pathlib.Path`. OpenMM treats non-string
inputs as file objects and calls `.read()`. The modular backend converts the
filename to `str`. The source parity gate permits exactly this conversion;
it does not omit production from comparison. The legacy backup remains unchanged.

The initial workflow harness incorrectly requested potential energy from a saved
OpenMM State, which contains positions/velocities but no energy field. This
harness check was corrected to validate saved positions and production-reported
energies. Restart and analysis validation then resumed from the already completed
initial runs. Initial logs, the harness failure, the legacy restart failure,
and the successful modular continuation are retained separately.

Analysis comparisons use the same trajectory for old and new analysis code.
Independently prepared solvent structures and stochastic trajectories are not
required to match byte-for-byte. Contact-analysis function parity is checked
structurally; external getcontacts/VMD execution was not exercised here.

## Environment

Python 3.11.14, OpenMM 8.4.0, NumPy 1.26.4, OpenFF Toolkit 0.18.0,
OpenMMForceFields 0.15.1, Espaloma 0.4.0, PyTorch 2.3.1, DGL 2.3.0,
RDKit 2025.09.4, MDTraj 1.11.0, and PDBFixer 1.12.0.

## Local evidence

Generated outputs remain under ignored `dev/`:

- `dev/refactor_baseline/manifest.json`: baseline hash, Git state, software versions,
  module hashes, and completed checks.
- `dev/refactor_baseline/mojoeMD.py` and `working-tree.patch`: starting script and
  pre-existing user changes, captured before extraction.
- `dev/refactor_baseline/modular-unit-final.log`, `ligand-parity.log`, and
  `package-build.log`: fast tests, actual ligand parameter comparisons, and build.
- `dev/refactor_workflow/modular/result.json`: successful workflow/analysis record.
- `dev/refactor_baseline/modular-workflow.log`, `modular-workflow-validation.log`,
  and `modular-workflow-final.log`: original execution, initial restart defect,
  and successful continuation after the fix.
- `dev/refactor_baseline/legacy-workflow.log` and `legacy-restart-baseline.log`:
  legacy initial execution and reproduced checkpoint filename defect.
- `dev/preparation_tests/modular_refactor_matrix/`: individual preparation
  configurations, logs, systems, diagnostics, result files, and suite summary.
- `dev/preparation_tests/modular_refactor_metals/`: both real copper-site preparations.
- `dev/preparation_tests/legacy_refactor_ff19_comparison/`: the corresponding
  legacy combined-module preparation also passed.

The source parity tests, small fixtures, and runners are visible to version
control. No Git commit was created. See [the migration record](../REFACTOR.md)
for commands and remaining acceptance gates before changing the default backend.
