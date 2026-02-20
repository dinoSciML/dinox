# dinox

`dinox` is a `JAX` implementation of Reduced Basis Derivative Informed Neural Operators, built for high performance in single-GPU environments where all training data fits in GPU memory.

The library is designed primarily for PDE learning workflows based on:

- `FEniCS 2019.1`
- Jacobians computed via `hippylib`
- Subspace methods provided by `bayesflux`

---

## Overview

`dinox` provides:

- Reduced basis neural operator architectures
- Derivative-informed training using PDE Jacobians
- GPU-accelerated implementations in `JAX` and `Equinox`
- Integration with FEniCS-discretized PDEs

---

## Getting Started

See the [hyperelasticity tutorial]([examples/DINO_Tutorial.ipynb](https://github.com/dinoSciML/dinox/blob/main/examples/DINO_Tutorial.ipynb)) for a complete walkthrough of the RB-DINO pipeline: problem setup, data generation, training with L2 vs H1 loss, and surrogate evaluation.

---

## Important: FEniCS & Hippylib Environment Required

`dinox` depends on `bayesflux[hippylib]`, which requires:

- hippylib
- FEniCS 2019.1

`FEniCS` has system-level dependencies and cannot be installed via pip alone.

You must first create a conda environment with FEniCS 2019.1 before installing dinox.

---

## Installation

### Prerequisites

- NVIDIA driver >= 525 (check with `nvidia-smi`)
- conda or mamba

> **Note on CUDA libraries:** You do **not** need to install CUDA Toolkit, cuDNN, or cuSPARSE via conda or your system package manager. The pip wheels for JAX and CuPy bundle their own CUDA 12 runtime libraries. Installing system CUDA alongside pip-bundled CUDA is the most common source of GPU detection failures.

### Step 1 — Create a FEniCS 2019.1 environment

```bash
conda create -n fenics-2019.1_env -c conda-forge fenics==2019.1.0 python=3.11
conda activate fenics-2019.1_env
```

### Step 2 — Fix `LD_LIBRARY_PATH` (critical for GPU)

A system-level or conda-set `LD_LIBRARY_PATH` pointing to a CUDA installation will conflict with the CUDA libraries bundled in the JAX and CuPy pip wheels, causing errors like `Unable to load cuSPARSE`.

```bash
unset LD_LIBRARY_PATH
```

### Step 3 — Install GPU-enabled JAX

```bash
pip install "jax[cuda12]" cupy-cuda12x nvidia-curand-cu12
```

### Step 4 — Install dinox

```bash
# With CuPy GPU support (recommended)
pip install dinox[cupy]

# Without CuPy
pip install dinox
```

### Step 5 — Verify GPU

```bash
python -c "import jax; print('JAX devices:', jax.devices())"
python -c "import cupy; print('CuPy GPU count:', cupy.cuda.runtime.getDeviceCount())"
```

You should see your NVIDIA GPU listed. If JAX shows only `CpuDevice`, check that `LD_LIBRARY_PATH` is unset (see Step 2).

---

## GPU Support

- Designed for single-GPU workflows where all data fits in GPU memory
- Requires CUDA 12-enabled JAX (`pip install "jax[cuda12]"`) — the pip wheel bundles its own CUDA runtime
- Optional CuPy arrays for GPU operations via `dinox[cupy]`
- Without GPU, CPU fallback is automatic

---

## Development

```bash
conda create -n fenics-2019.1_env -c conda-forge fenics==2019.1.0 python=3.11
conda activate fenics-2019.1_env

pip install "jax[cuda12]" cupy-cuda12x nvidia-curand-cu12
unset LD_LIBRARY_PATH  # or use the permanent conda hook above
pip install -e ".[dev]"
```

This installs:

- dinox (editable)
- development tools (pytest, black, flake8, isort)
- bayesflux[hippylib]
- hippylib
- all required JAX dependencies

---

## Requirements

- Python >= 3.10
- FEniCS 2019.1 (via conda)
- JAX >= 0.7.0 (for GPU: `pip install "jax[cuda12]"`)
- NVIDIA driver >= 525 (for GPU)
- Optional: CuPy for GPU array operations (`pip install dinox[cupy]`)

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Unable to load cuSPARSE` | `unset LD_LIBRARY_PATH` before running Python |
| `No such file: libcurand.so` | `pip install nvidia-curand-cu12` |
| JAX shows only `CpuDevice` | Ensure `jax[cuda12]` was installed (not just `jax`) and `LD_LIBRARY_PATH` is unset |
| `nvidia-smi` not found | Install or update NVIDIA driver (>= 525) |
| JAX/CuPy CUDA version conflict | Do **not** `conda install cudatoolkit` — let pip wheels provide CUDA |

---

## Repository

- Homepage: https://github.com/dinoSciML/dinox
- Repository: https://github.com/dinoSciML/dinox
