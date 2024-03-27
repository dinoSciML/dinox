# dinox
Implementation of **Derivative Informed Neural Operators** in `jax`. Build for fast performance in single GPU environments-- and specifically where _all-data-can-fit-in-gpu-memory_. In the future, this code will be generalized for the setting in which one has multiple GPUs and would like to take advantage. It will also be generalized to account for big-data (where not all samples can fit in gpu or cpu memory). 

# Installation
Create a brand new environment. Use `mamba` in place of `conda` if you can.
```
conda env create -f environment.yml
poetry install
```
# Running dinox
```
python -m dinox -network_name "<name_to_save_network_as>" -data_dir "<location_of_jacobian_enriched_training_data>"
```
# Examples
Currently, one can run in each application folder 

```
python generate_data.py
```

**if one has `HIPPYFLOW_PATH` and `HIPPYLIB_PATH` environmental variables set correctly and one has both of those libraries.** TODO: DO away with this. For the paper, we ought to provide two options.
1) Directly download our training data (as generated from hippylib/flow with directions on exactly how it was generated
2) Install hippylib/flow and user our applications folders to generate them yourself (at your own risk)

Currently, assumes data is stored in "m_data.npy", "q_data.npy", "J_data.npy". Will generalize soon for CLI passed in npy filenames, and for memmapped numpy arrays.. 


# Notes on why we require these packages:
- `cupy` - for rapid permuting of data on GPUs
- `kvikio` - for interfacing with NVIDIA GPU Direct Storage (GDS) for loading data directly to GPU, skipping the CPU
- `opt_einsum` - for order-optimized tensor contraction
- `equinox` - Dinox is primarily build off of equinox and is therefore fully jax compatible. Most of dinox are simply lightweight utilities for dealing with mean H1 loss training of nerual networks with data that is enriched with Jacobians ($`X, Y, dY/dX`$)
- `optax` - we use optax for optimization, though any neural network optimization library can be used. We make choices primarily for speed.

## Need to generalize this to figure out the actual minimal requirements in terms of cuda, jax versions, and kvikio. The main tricky parts are which versions of jax/kvikio/cudatoolkit/cuda-nvcc/cudnn work together well. For now, only want to restrict to python>=3.10
## Let me know if anyone has depenency resolution issues.
