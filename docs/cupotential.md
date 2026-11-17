# Optional cuPotential production acceleration

Select `--acceleration cupotential` with the modular backend on a compatible
Linux NVIDIA GPU node. The default is `openmm`. Preparation and equilibration
use the existing OpenMM implementation; production uses cuPotential's nonbonded
force, LangevinMiddle integrator and isotropic Monte Carlo barostat.

This is an experimental integration. Local adapter tests and a real stock CPU
production/restart pass on the Mac; cuPotential execution, accuracy, performance,
and GPU checkpoint continuation must still be validated on the cluster. There is
no NVIDIA CUDA GPU on this development machine. No speedup has been measured here.

## Cluster requirements and installation

Use a separate environment from your working installation. This adapter requires
[NVIDIA's 0.3.5 release](https://github.com/NVIDIA/cuPotential/releases/tag/v0.3.5).
Its [binary metadata](https://github.com/NVIDIA/cuPotential/blob/v0.3.5/plugins/openmm/write_package_metadata.py)
requires OpenMM **8.4.x**. The release artifacts currently use **CUDA 13.4**;
the published wheel is **Python 3.12, Linux x86-64, glibc >= 2.38**. These
requirements differ from the CUDA 13.1 example in NVIDIA's introductory blog.
Check your cluster's GPU driver, CUDA runtime and Linux/container compatibility.
GPU access alone is insufficient. `mojoemd doctor --acceleration cupotential`
creates an actual CUDA Context and takes a step to catch runtime load failures.

On an existing full Python 3.12/OpenMM 8.4 environment satisfying those platform
requirements, install the published wheel:

```bash
python -m pip install https://github.com/NVIDIA/cuPotential/releases/download/v0.3.5/nv_openmm_plugin-0.3.5-cp312-cp312-manylinux_2_38_x86_64.whl
```

For the existing Python 3.11 cluster environment, build matching Python bindings
from source. Start from a clone of the working cluster environment, preserving
its ligand/analysis dependencies. These are Linux commands, not Mac commands:

```bash
conda create -n mojoeMD-cupotential --clone mojoeMD
conda activate mojoeMD-cupotential
mamba install -c conda-forge openmm=8.4.0 cuda-version=13.4 cuda-toolkit=13.4 cmake make gcc_linux-64=13 gxx_linux-64=13 swig
conda deactivate
conda activate mojoeMD-cupotential
python -m pip install 'pybind11>=2.12' nvidia-mathdx

git clone --branch v0.3.5 --depth 1 https://github.com/NVIDIA/cuPotential.git
cd cuPotential
python - <<'PY'
from pathlib import Path
p = Path('plugins/openmm/run_build')
text = p.read_text()
old = 'OPENMM=/opt/openmm'
assert old in text, 'Inspect the upstream build helper before adapting it'
p.write_text(text.replace(old, 'OPENMM="${CONDA_PREFIX:?Activate the conda environment first}"'))
PY
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CUDAHOSTCXX="${CXX:?Activate the environment with the conda compiler installed}"
./plugins/openmm/run_build
python -m pip wheel --no-deps --no-build-isolation --wheel-dir plugins/openmm/dist ./plugins/openmm
python -m pip install plugins/openmm/dist/*.whl
```

The upstream helper downloads the matching cuPotential runtime. A source build
of the Python bindings alone cannot fix an incompatible precompiled runtime or
old system libraries: use a compatible cluster container/runtime if needed.
NVIDIA's build instructions and release metadata are authoritative for these
early-access binaries. Install mojoeMD in that environment from this branch:

```bash
cd /path/to/mojoeMD
git fetch origin
git switch cupotential
python -m pip install --no-deps -e .
```

## Run and restart

Use a new output directory and real input paths:

```bash
mojoemd doctor --acceleration cupotential
mojoemd run -c examples/cupotential_protein.json
mojoemd produce -c examples/cupotential_protein.json --restart
```

Or enable it for a configuration via the CLI:

```bash
mojoemd run -c my_system.json --output-dir runs/my_cupotential --acceleration cupotential
mojoemd produce -c my_system.json --output-dir runs/my_cupotential --acceleration cupotential --restart
```

The JSON setting is `"acceleration": "cupotential"`. CLI values override the JSON.
`--backend` still selects `modular` versus `legacy`; acceleration requires
`modular`. The original `python mojoeMD.py` engine does not support it.
Preparation-only jobs can use the same JSON on a Mac or CPU host without loading
the plugin; cuPotential is required when production is enabled.

`prepared.pdb`, `system.xml`, `equilibrated.pdb`, `equilibrated_system.xml` and
`equilibrated.xml` stay in standard OpenMM format. The production System is
converted in memory before Context creation. It replaces the existing isotropic
barostat rather than adding a second one, and transfers COM removal to the plugin
integrator using the original removal frequency. Other forces, restraints, particle
parameters and input files remain intact. The source cutoff, switching settings
and supported accuracy settings are passed to the plugin without retuning.

`production_backend.json` records the acceleration mode and OpenMM/plugin
versions. Restart requires the same backend and, for cuPotential, recorded software
versions. Keep the same GPU/platform setup as well; OpenMM binary checkpoints
are hardware/software dependent. A stock checkpoint cannot be loaded with the
plugin integrator. Older stock runs without this record remain restartable with
`--acceleration openmm`.

To try cuPotential on a previously equilibrated stock system, copy the three
`equilibrated*` files into a new output directory and start production **without**
`--restart`, using the same timestep settings as the original configuration.
This starts from the portable equilibrated XML state, not a production checkpoint.

## Initial supported scope

- One standard `NonbondedForce`, PME, rectangular periodic box larger than twice
  the cutoff, Ewald error tolerance 5e-4 and automatic PME parameters.
- Standard Amber bonded forces, at most one isotropic Monte Carlo barostat and
  one COM remover, plus `CustomBondForce` / `CustomExternalForce` restraints.
- A single CUDA device selected by the scheduler (`CUDA_VISIBLE_DEVICES`).

The adapter rejects LJPME, triclinic boxes, parameter offsets, split direct/reciprocal
force groups, CHARMM/custom nonbonded forces, anisotropic/membrane barostats and
other unhandled forces. Some of these are supported by newer upstream APIs; they
are deliberately outside this first adapter's validation scope. Membrane and
complex glycan/NCAA/metal-site dynamics have not been validated with cuPotential.
Benchmark your actual system before assuming the published speedup applies.
Keep normal timesteps/cutoffs and validate energy/force errors and ensemble
behavior before changing scientific settings.

## Validation on an allocated GPU node

```bash
mojoemd doctor --acceleration cupotential
MOJOEMD_CUPOTENTIAL_INTEGRATION=1 python -m unittest discover -s tests -p test_cupotential_integration.py -v
python tests/workflow_regression.py --backend modular --acceleration cupotential --output dev/cupotential-workflow-new
```

The GPU integration test compares stock and accelerated water forces/energies,
round-trips plugin XML, runs NPT steps, and tests a checkpoint in a freshly rebuilt
Context. Its 1% relative-error thresholds are regression gates, not universal
accuracy guarantees. The workflow runner retains all 130,000 stock equilibration
steps on CPU and then tests two 50 ps CUDA production segments, checkpoint
continuation and analysis. Both output directories and raw logs should be retained
with GPU/driver details. These commands have not been run on this Mac.

An example scheduler script (adjust resources and paths for your cluster):

```bash
#!/bin/bash
#SBATCH --job-name=mojoe-cupotential
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mojoeMD-cupotential
cd /path/to/mojoeMD
export OPENMM_CPU_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mojoemd doctor --acceleration cupotential
mojoemd run -c my_system.json --output-dir runs/my_cupotential --acceleration cupotential
```
