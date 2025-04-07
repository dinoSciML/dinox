# dinox
Implementation of **Derivative Informed Neural Operators** in `jax`. Build for fast performance in single GPU environments-- and specifically where _all-data-can-fit-in-gpu-memory_. In the future, this code will be generalized for the setting in which one has multiple GPUs and would like to take advantage. It will also be generalized to account for big-data (where not all samples can fit in gpu or cpu memory) -- Probably via memmapping. 

# Installation
Create a brand new environment. Use `mamba` in place of `conda` if you can. (i.e. run the first line below). The assumption is that conda is already installed on your machine.

If one has access to an NVIDIA gpu, use gpu_environment.yml, otherwise use cpu_environment.yml, which will install the dependencies for the code, but the code will not be as performant, since the library is a GPU-forward library.
```
conda install -c conda-forge mamba

mamba env create -f <gpu, cpu>_environment.yml
poetry install
```
# Running dinox
```
python -m dinox -network_name "<name_to_save_network_as>" -data_dir "<location_of_jacobian_enriched_training_data>"
```
# Examples


# Note, the codebase needs to be generalized to work generally on CPUs. It also does not fully work on Apple Silicon (jax-metal has limitations)
# Notes on why we require these packages:
- `cupy` - for rapid permuting of data on GPUs
- `kvikio` - for interfacing with NVIDIA GPU Direct Storage (GDS) for loading data directly to GPU, skipping the CPU
- `equinox` - Dinox is primarily build off of equinox and is therefore fully jax compatible. Most of dinox are simply lightweight utilities for dealing with mean H1 loss training of nerual networks with data that is enriched with Jacobians ($`X, Y, dY/dX`$)
- `optax` - we use optax for optimization, though any neural network optimization library can be used. We make choices primarily for speed.

## Need to generalize this to figure out the actual minimal requirements in terms of cuda, jax versions, and kvikio. The main tricky parts are which versions of jax/kvikio/cudatoolkit/cuda-nvcc/cudnn work together well. For now, only want to restrict to python>=3.10
## Let me know if anyone has depenency resolution issues.
