# Example preparation and modular-default validation: 2026-11-06

All **11 numbered example profiles passed preparation** with explicitly mapped
local inputs. The modular backend is now the default for `mojoemd`,
`python -m mojoemd`, and all public simulation API helpers. Select the independent
original engine with `--backend legacy` / `backend="legacy"`. Explicit
`--backend modular` remains valid. Direct `python mojoeMD.py` remains legacy.

## Results

| Example template | Result | Prepared atoms |
| --- | --- | ---: |
| `01_vanilla_protein.json` | Passed | 94,081 |
| `02_protein_ligand.json` | Passed | 95,176 |
| `03_glycoprotein.json` | Passed | 121,348 |
| `04_protein_ncaa.json` | Passed | 177,751 |
| `05_glycoprotein_ligand.json` | Passed | 146,160 |
| `06_glycoprotein_ncaa.json` | Passed | 121,348 |
| `07_protein_ligand_ncaa.json` | Passed | 325,574 |
| `08_full_combo_glyco_ligand_ncaa.json` | Passed | 146,423 |
| `09_prepare_only_debug_glycoprotein.json` | Passed | 120,863 |
| `10_metal_coordinated_protein_ligand.json` | Passed | 95,190 |
| `11_membrane_ligand.json` | Passed | 316,220 |

Every successful case passed topology/particle consistency, finite-coordinate,
periodic-box, retained-chemistry, serialization, and finite initial energy/force
checks. Additional checks cover glycan covalent links, NCAA peptide connectivity,
metal restraints, ligand composition, and membrane lipids when applicable.

No pipeline equilibration or production ran: the runner forces `prepare` only,
patches those other stages to fail if invoked, checks logs, and forbids trajectory
and checkpoint artifacts. OpenMM's internal preparation relaxation is retained.

After promotion, **27 unit checks passed**, with one opt-in integration wrapper
skipped because preparation was executed separately. A real default-CLI preparation
passed with the legacy loader patched to fail if called. The rebuilt wheel matches
all current Python sources; isolated wheel imports verify modular defaults for all
four public simulation helpers, the CLI default, and explicit legacy availability.

## Input mapping and limits

These are **resolved-input runs, not unchanged execution of the template JSONs**.
Original examples, structures, parameters, and the legacy engine were not edited.
The manifest records original JSON, exact substitutions, input hashes, versions,
source hashes, and output policy. Each case also retains its effective config,
input mapping, logs, prepared PDB, serialized system, and applicable diagnostics.

- Examples 1, 2, and 10 use the full copper-containing `dev/inputs/protein.pdb`.
  Examples 2 and 10 use synthetic ethanol for positive software coverage.
- Examples 3, 5, 6, 8, and 9 use the full local 33-sugar glycoprotein/AIB structure
  in `dev/refactor_retests/ncaa_glycan/`. With NCAA mode disabled, ordinary repair
  may canonicalize the AIB residue; with NCAA enabled, AIB is preserved and its
  matching XML is used.
- Examples 4 and 7 resolve the generic NCA placeholders to CIR/HYP in the full
  `dev/refactor_retests/ncaa/5MU0.pdb`, with the existing matching CIR parameters.
- Positive non-membrane ligand profiles use the known-valid synthetic ethanol
  fixture. These are software combination tests, not biological binding models.
- Example 11 uses the **matching** full membrane receptor and ligand from
  `dev/membrane/protein.pdb` and `dev/membrane/ligand.sdf`, with original coordinates.
- Example 9 retains debug target index 527. It is absent from this local structure;
  the diagnostic correctly reports that the target was not found. Debug PDB and
  template diagnostics were written successfully.
- The distinct `dev/inputs/ligand.sdf` remains malformed: six aromatic bonds are
  encoded outside rings. Its expected rejection is recorded separately in
  `malformed_ligand.json`; no bond-order or protonation guesses were made.

This validates preparation and default selection, not complex-system equilibration,
long-term dynamics, biological suitability of synthetic ligand combinations, or
scientific convergence. The earlier small-system production/restart and analysis
validation remains documented in the November 4 record. The original legacy engine
still has the known checkpoint filename bug; modular contains the tested fix.

## Evidence and reproduction

All generated logs and simulation outputs are in `dev/modular_prep_check/`:

- `manifest.json`, `summary.json`, and `runner.log`.
- One directory per numbered example, including `config.json`, `input_mapping.json`,
  `console.log`, `result.json`, and `output/`.
- `malformed_ligand.json`, `unit-tests.log`, and `default-cli-smoke.log`.
- `wheels/`, `wheel-build.log`, and `wheel-import.log`.

Run the checks into a **new** output directory using the existing environment:

```bash
python tests/example_preparation.py --output dev/modular_prep_check_new --jobs 2
python -m unittest discover -s tests -v
```

`examples/` remains a set of templates. Supply real input paths/NCAA identities or
use a retained effective config and a new output directory. The existing Python,
OpenMM, PyTorch/DGL/Espaloma environment and scientific protocols were unchanged.
