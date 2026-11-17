# cuPotential adapter: local validation, 2026-11-17

Implemented on branch `cupotential`, based on merged `main`. cuPotential is
opt-in for modular production; preparation, equilibration and the default stock
production dynamics are retained. The adapter requires OpenMM 8.4.x and NVIDIA's
nv-openmm-plugin 0.3.5. No cuPotential binaries or third-party source are vendored.

The API was checked against NVIDIA's public 0.3.5 release and source, including
its force conversion, unit-aware factories, serialization and checkpoint tests,
and wheel requirements. Release assets use CUDA 13.4, whereas the introductory
blog shows a CUDA 13.1 setup. Installation must follow the selected release's
actual runtime/driver/Python/Linux requirements.

## Executed on the Apple Silicon development machine

- Unit discovery: **53 tests**, **51 passed**, **2 skipped**. The skipped tests
  are the existing opt-in preparation wrapper and the new real CUDA integration
  test. Mocked adapter tests do not establish GPU execution or accuracy.
- Source parity: complete default preparation/equilibration/production paths
  still match the documented legacy behavior, after narrowly normalizing the
  new config option, acceleration branch and backend-record calls.
- Real stock production: two 1,000-step CPU segments at 2 fs, starting from the
  retained small-system equilibrated state. The second invocation loads the
  first checkpoint. Reloading the final checkpoint confirms step **132,000**
  (130,000 equilibration plus 2,000 new production steps). Backend recorded as
  `openmm`; the existing source state/system were copied, not modified.
- Installed `mojoemd doctor`: healthy.
- `mojoemd doctor --acceleration cupotential`: fails explicitly on this Mac with
  the required Linux x86-64/NVIDIA GPU diagnostic, as expected.
- Python compilation and `git diff --check`: passed.

Local raw evidence is retained under `dev/cupotential-local-validation/`:
`unit-tests.log`, `stock-production.log`, `stock-production-result.json`,
`stock-production/`, `doctor-stock.log`, and `doctor-cupotential-mac.log`.

## Still required on the cluster

Run the [GPU tests and workflow](../docs/cupotential.md#validation-on-an-allocated-gpu-node)
on compatible hardware. They check water energy/force agreement, plugin XML
roundtrip, fresh-Context NPT checkpoint continuation, and the full pipeline's
production/restart/analysis handoffs. None of those cuPotential GPU checks have
been executed here. No speedup, ensemble equivalence for user systems, or
scientific convergence is claimed. Complex systems and unsupported force/box
features need separate validation before extending the conservative adapter.
