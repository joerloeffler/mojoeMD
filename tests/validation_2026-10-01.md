# Preparation validation: 2026-10-01

The current suite's 127 cases passed across the full run and targeted copper-site
reruns: 112 real preparations and 15 expected configuration rejections. The
initial full run had two copper-site input failures; these are preserved and
explained below. The final fast unittest run passed 17 tests and skipped the
opt-in integration wrapper, whose underlying runner was executed separately.

Environment: OpenMM 8.4.0, NumPy 1.26.4, OpenFF Toolkit 0.18.0,
OpenMMForceFields 0.15.1, Espaloma 0.4.0, PyTorch 2.3.1, DGL 2.3.0,
and RDKit 2025.09.4.

| Preparation coverage | Result |
| --- | --- |
| Complete small matrix, including all exposed protein models and expected rejections | 121/121 passed |
| 9UTB membrane + ligand, SMIRNOFF / Espaloma | Passed; 316,094 / 316,634 atoms |
| Full 33-sugar glycoprotein + membrane + ligand + AIB, SMIRNOFF / Espaloma | Passed; 164,635 / 164,617 atoms |
| Copper-site protein + valid test ligand, SMIRNOFF / Espaloma | Passed; 95,352 / 95,176 atoms |

The supplied local 9UTB files contain no modeled glycans. Glycosylation coverage
comes from the separate linked-glycan fixtures; it is not inferred from the
identity of the 9UTB protein.

## Issues found and changes

- Legacy Amber protein XML already contains ions. Pairing it with Amber14 water
  caused duplicate sodium templates during real system creation. Legacy models
  now default to standalone `tip3p.xml` and reject the duplicating water files.
- OpenMM's `getUnmatchedResidues()` does not apply CHARMM patches, so the old
  precheck rejected ordinary termini. CHARMM validation now uses `createSystem()`
  without a simulation context to check patched templates and parameters.
- The first small membrane fixtures had a membrane center inconsistent with
  their translated coordinates. The test center now matches the fragments.
  Membrane output checks also account for PDB shortening `POPC` to `POP`.
- The original `dev/inputs/ligand.sdf` has six non-ring bonds encoded as aromatic.
  OpenFF conversion produced radicals and SMIRNOFF charge assignment failed.
  Ligand preparation now rejects that encoding clearly before conversion. The
  original file is unchanged and a copy is retained as a negative unit fixture.
  A positive unit test verifies that normal aromatic rings and implicit
  hydrogens remain accepted.
- Positive copper-site stress tests now use the known-valid small ligand fixture.
  These tests verify copper retention and restraints involving the copper site;
  they do not claim successful preparation of the malformed original ligand.

The full run's initial result was 125/127 passed with two failures on that
original copper-site ligand. Two targeted positive copper-site reruns passed.
The consolidated result checks that each retained case config exactly matches
the current suite. Valid test and 9UTB ligands were also accepted by the new
input validation. The changed input precheck does not alter their parameters.

## Local evidence

The following artifacts are under ignored `dev/` and remain available in this
workspace; they are not committed simulation outputs:

- `dev/preparation_tests/full_regression/summary.json`: complete initial run,
  including both original copper-ligand failures and all four large successes.
- `dev/preparation_tests/metal_valid_ligand/summary.json`: two successful
  copper-site preparations with the valid test ligand.
- `dev/preparation_tests/validated_results.json`: consolidated current-suite
  results, versions, links to case artifacts and preserved original failures.

Only preparation ran. OpenMM's hydrogen addition and membrane packing include
their own preparation relaxation. No pipeline equilibration or production stages
ran, and no trajectories or checkpoints were produced. The retained-chemistry,
serialization and finite-energy checks do not establish scientific equilibration.

See [test instructions](README.md) for reproducing the suites and input requirements.
