# dinox
Implementation of **Derivative Informed Neural Operators** in `jax`. Build for fast performance in single GPU environments-- and specifically where _all-data-can-fit-in-gpu-memory_. In the future, this code will be generalized for the setting in which one has multiple GPUs and would like to take advantage. It will also be generalized to account for big-data (where not all samples can fit in gpu or cpu memory). 

# TODO: install directions...
## currently tested on:

Create a brand new environment. Use `mamba` in place of `conda` if you can.
```
conda create -n dinox_env -c nvidia -c conda-forge -c rapidsai python=3.10 cupy=13.0.0 cudnn=8.8.0.121 cuda-nvcc=12.3.107 cudatoolkit=11.8.0 jax=0.4.25 kvikio=24.02.01 optax=0.1.9 opt_einsum=3.3.0
#The below can be updated without any problems
conda update optax 
conda update opt_einsum
conda update cupy
pip install equinox
```

# Notes on why we require these packages:
- `cupy` - for rapid permuting of data on GPUs
- `kvikio` - for interfacing with NVIDIA GPU Direct Storage (GDS) for loading data directly to GPU, skipping the CPU
- `opt_einsum` - for order-optimized tensor contraction
- `equinox` - Dinox is primarily build off of equinox and is therefore fully jax compatible. Most of dinox are simply lightweight utilities for dealing with mean H1 loss training of nerual networks with data that is enriched with Jacobians ($`X, Y, dY/dX`$)
- `optax` - we use optax for optimization, though any neural network optimization library can be used. We make choices primarily for speed.

## Need to generalize this to figure out the actual minimal requirements in terms of cuda, jax versions, and kvikio. The main tricky parts are which versions of jax/kvikio/cudatoolkit/cuda-nvcc/cudnn work together well. For now, only want to restrict to python>=3.10
## Let me know if anyone has depenency resolution issues.
