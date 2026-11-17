# Modular branch and Apple Silicon validation: 2026-11-17

Reviewed the `modular` branch and repaired the existing native arm64 `mojoeMD`
environment on this Mac. Repository changes are separate from installed-package
changes; `requirements-macos-espaloma.txt` records the repair overlay.

## Logic fixes

- `doctor` now loads the default modular engine rather than the legacy engine.
- It checks PyTorch/DGL versions, actually creates a DGL sparse adjacency matrix,
  and checks whether OpenFF can use AmberTools on the active `PATH`.
- Both engines guard the optional `openff` parent namespace before searching for
  `openff.toolkit`, so a core-only installation can import the ligand module.
- Removed `amber19/protein.ff19ipq.xml` from both engines' advertised choices:
  upstream withdrew this incomplete model. Remaining force-field loading tests pass.
- API results find production segments numerically, including segment 1000.
  Legacy/modular chemistry function parity remains checked.

## Espaloma repair

Espaloma, DGL and QCPortal were initially absent. The existing conda pins did not
solve on native Apple Silicon Python 3.11. The working stack uses official native
PyPI wheels plus the official Espaloma source release:

| Component | Version |
| --- | --- |
| Python | 3.11.16 |
| OpenMM / OpenMMForceFields | 8.6.1 / 0.16.0 |
| OpenFF Toolkit | 0.18.0 |
| Espaloma package / cached parameter model | 0.4.0 / 0.3.2 |
| PyTorch / DGL | 2.0.1 / 1.1.3 |
| NumPy / MDTraj | 1.26.4 / 1.11.0 |
| JAX / jaxlib | 0.4.35 / 0.4.35 |
| PyTorch Lightning | 2.0.9 |
| QCPortal / QCElemental | 0.63 / 0.29.0 |

DGL's native wheel includes sparse libraries for PyTorch 2.0.1, but not 2.3.1.
Imports alone did not expose this mismatch; real Espaloma parameterization did.
NumPy-compatible MDTraj and JAX were also required. Conda-managed JAX packages
were removed before installing their replacements because pip could not uninstall
packages without RECORD metadata. Existing AmberTools helper requirements were
resolved with Biopython 1.85 and PDB2PQR 3.8.0.

`pip check` reports no dependency-version conflicts, but prints a platform message
for torch 2.0.1: its official arm64 wheel's internal WHEEL tag incorrectly says
x86_64. `file` confirms both `_C.cpython-311-darwin.so` and `libtorch_cpu.dylib`
are arm64. DGL sparse operations and both parameterizers execute successfully.
No installed package source or wheel metadata was patched.

## Completed checks

- Installed `mojoemd doctor`: healthy with the environment activated.
- Unit suite: 39 tests run, 38 passed, one opt-in integration wrapper skipped.
- Real ligand parity: SMIRNOFF and Espaloma both produce identical serialized
  systems and coordinates from legacy and modular parameterizers.
- All 11 numbered profiles: preparation passed through both the API and CLI.
  CLI cases also pass retained-chemistry, particle-count, serialization, finite
  coordinate/energy/force, membrane, glycan, NCAA, and metal-restraint checks.
- Malformed aromatic-bond ligand: rejected as expected.
- Small modular workflow: passed all 130,000 existing equilibration steps,
  50 ps production plus 50 ps checkpoint continuation on CPU at 2 fs, and
  combined-trajectory analysis. Each segment contains one saved frame; reported
  times advance from 300 to 350 ps. Analysis CSVs match the legacy analysis on
  identical combined input. This verifies execution/handoffs, not convergence.

The 11 example templates were explicitly mapped to committed small software
fixtures, including synthetic ethanol, AIB and a synthetic Cu/ASN contact. Large
development inputs used on the previous machine are absent here. This does not
revalidate those large structures, biological binding, membrane orientation,
metal-site chemistry, or 100 ns dynamics. The original templates are unchanged.
The debug example retains its original target index, which is absent in the small
fixture and is reported by its diagnostics.

## Reproduce

```bash
conda activate mojoeMD
mojoemd doctor
python -m unittest discover -s tests -v
python tests/ligand_parity.py
python tests/example_preparation.py --inputs fixtures --frontend cli --output dev/example-check-new --jobs 2
python tests/workflow_regression.py --backend modular --output dev/workflow-check-new
```

Generated artifacts are ignored development outputs:

- `dev/mac-example-cli-20261117/`: all 11 CLI cases, manifest, exact input
  mappings/hashes, summary, logs, environment freeze and software versions.
- `dev/mac-example-check-20261117-fixed/`: all 11 API cases and their artifacts.
- `dev/mac-workflow-20261117/`: the small-system dynamics regression.

The README distinguishes `mojoemd doctor` diagnostics, preparation-only fixture
checks, and actual configured CLI simulations with user-supplied real inputs.
