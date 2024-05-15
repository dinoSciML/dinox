# This file is part of the dinox package
#
# dinox is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2 of the License, or any later version.
#
# dinox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Joshua Chen and Tom O'Leary-Roseberry
# Contact: joshuawchen@icloud.com | tom.olearyroseberry@utexas.edu

__all__ = [
    "sub_dict",
    "create_array_permuter",
    "save_to_pickle",
    "slice_data",
    "load_data_disk_direct_to_gpu",
    "split_training_testing_data",
]

import pickle
from os import makedirs
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import cupy as cp
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.dlpack import from_dlpack as jax_from_dlpack
from jax.dlpack import to_dlpack as jax_to_dlpack
from jax.lax import dynamic_slice_in_dim as jittable_slice
from kvikio.numpy import LikeWrapper


def sub_dict(*, super_dict: Dict[Any, Any], keys: Iterable[Any]) -> Dict[Any, Any]:
    """
    Extracts a subset of key-value pairs from a dictionary based on a specified iterable of keys.

    Parameters
    ----------
    super_dict : Dict[Any, Any]
        The dictionary from which to extract the subset.
    keys : Iterable[Any]
        An iterable containing the keys for which key-value pairs should be extracted
        from the super dictionary.

    Returns
    -------
    Dict[Any, Any]
        A dictionary containing only the key-value pairs from the original dictionary
        (`super_dict`) that match the keys specified in `keys`.

    Examples
    --------
    >>> super_dict = {'a': 1, 'b': 2, 'c': 3}
    >>> keys = ['a', 'c']
    >>> sub_dict = sub_dict(super_dict=super_dict, keys=keys)
    >>> print(sub_dict)
    {'a': 1, 'c': 3}

    Notes
    -----
    This function does not modify the original dictionary but returns a new dictionary
    that includes only the specified keys. If a key from the `keys` iterable does not exist
    in `super_dict`, it will not be included in the returned dictionary, and no error will
    be raised. Ensure the `keys` iterable includes only valid keys present in `super_dict`
    to avoid missing entries in the output.
    """
    return {k: super_dict[k] for k in keys if k in super_dict}


# Use jax permutation, which uses a slower algorithm
# def create_array_permuter(N) -> Callable:
#     indices = jnp.arange(N)

#     @jax.jit
#     def permute_arrays(
#         X: jax.Array,
#         Y: jax.Array,
#         dYdX: jax.Array,
#         Y_norms: jax.Array,
#         dYdX_norms: jax.Array,
#         key: jr.PRNGKey,
#     ) -> Tuple[jr.PRNGKey, Tuple[jax.Array]]:
#         (key, subkey) = jr.split(key)
#         perm = jr.permutation(subkey, indices)
#         return key, (X[perm], Y[perm], dYdX[perm], Y_norms[perm], dYdX_norms[perm])

#     return permute_arrays

# Use cupy permutation, which uses a faster algorithm


def create_arrays_permuter(N: int, cp_random_seed: int = None) -> Callable:
    """
    Creates a callable that permutes arrays using a potentially fixed random seed. The
    permute function generated can reorder arrays based on random permutation indices
    created by CuPy and applied to JAX arrays.

    Parameters
    ----------
    N : int
        The size of the array to generate permutation indices for.
    cp_random_seed : int, optional
        An optional seed to use for CuPy's random number generator, which ensures
        reproducibility of the permutation pattern.

    Returns
    -------
    Callable
        A function that, when called with JAX arrays, returns the permuted arrays.
        This function uses CuPy to generate permutation indices which are then used
        to reorder JAX arrays.

    Notes
    -----
    This function leverages CuPy to handle the generation of random permutation indices
    and JAX's DLPack capabilities to interchange data between CuPy and JAX efficiently.
    The seed setting impacts only the permutation pattern, making subsequent calls to
    the returned function produce the same permutation if the same seed is used.

    Example
    -------
    >>> N = 100
    >>> permute_arrays = create_array_permuter_flat(N, cp_random_seed=42)
    >>> X = jnp.array([...])
    >>> fX_dYdX = jnp.array([...])
    >>> X_permuted, fX_dYdX_permuted = permute_arrays(X, fX_dYdX)
    """
    indices = cp.arange(N)
    if cp_random_seed is not None:
        cp.random.seed(cp_random_seed)

    def permute_arrays(*arrays: Iterable[jax.Array]) -> Tuple[jax.Array, ...]:
        # Create a permutation of the indices
        perm = cp.random.permutation(indices)

        # Use DLPack to transfer data between CuPy and JAX and apply permutation
        return (
            jax_from_dlpack(cp.from_dlpack(jax_to_dlpack(arr))[perm]) for arr in arrays
        )

    return permute_arrays


@eqx.filter_jit
def slice_flat_data(
    X: jax.Array, fX_dfXdX: jax.Array, batch_size: int, end_idx: int
) -> Tuple[int, jax.Array, jax.Array]:
    """
    Slices arrays X and fX_dfXdX for the given batch size starting from a specified index,
    commonly used for batching operations during data processing.

    Parameters
    ----------
    X : jax.Array
        The array of input features to be sliced.
    fX_dfXdX : jax.Array
        The array of concatenated output features and their jacobians to be sliced.
        This array is expected to be flat, typically containing output values concatenated
        with their flattened jacobians.
    batch_size : int
        The number of samples to include in each slice.
    end_idx : int
        The starting index from where the slice should begin.

    Returns
    -------
    Tuple[int, jax.Array, jax.Array]
        - An integer indicating the updated end index after slicing, which can be used
          for subsequent slicing operations.
        - A sliced portion of the X array corresponding to the specified batch size.
        - A sliced portion of the fX_dfXdX array corresponding to the same batch size.

    Notes
    -----
    This function is designed for use in data preprocessing or training loops where
    data needs to be processed in segments or batches. It returns segments of the
    input and output data starting from `end_idx` and extending up to `end_idx + batch_size`.
    No modifications are made to the original data arrays.

    Example
    -------
    >>> X = jnp.array([...])  # some input features
    >>> fX_dfXdX = jnp.array([...])  # output features and jacobians, flattened
    >>> batch_size = 100
    >>> end_idx = 0
    >>> new_end_idx, X_batch, fX_dfXdX_batch = slice_flat_data(X, fX_dfXdX, batch_size, end_idx)
    >>> print(new_end_idx, X_batch.shape, fX_dfXdX_batch.shape)
    """
    return (
        end_idx + batch_size,
        jittable_slice(X, end_idx, batch_size),
        jittable_slice(fX_dfXdX, end_idx, batch_size),
    )


@eqx.filter_jit
def slice_data(
    X: jax.Array, fX: jax.Array, dfXdX: jax.Array, batch_size: int, end_idx: int
) -> Tuple[int, jax.Array, jax.Array, jax.Array]:
    """
    Slices arrays X, fX, and dfXdX from the specified starting index (end_idx) up to
    the specified batch size, incrementing the index for subsequent operations.

    Parameters
    ----------
    X : jax.Array
        Input features array to be sliced.
    fX : jax.Array
        Output features array corresponding to X, to be sliced.
    dfXdX : jax.Array
        Derivative of fX with respect to X, to be sliced.
    batch_size : int
        The number of samples to include in the slice.
    end_idx : int
        The starting index from where the slicing should begin.

    Returns
    -------
    Tuple[int, jax.Array, jax.Array, jax.Array]
        A tuple containing:
        - The updated end index after adding the batch size, which can be used for the next slice.
        - The sliced segment of the X array.
        - The sliced segment of the fX array.
        - The sliced segment of the dfXdX array.

    Notes
    -----
    This function is useful in batching scenarios where data needs to be processed
    in smaller segments, typically during training loops in machine learning models.
    The function directly supports the operation of slicing without modifying the
    original data.

    Example
    -------
    >>> import jax.numpy as jnp
    >>> X = jnp.array([...])  # assume some data
    >>> fX = jnp.array([...])
    >>> dfXdX = jnp.array([...])
    >>> batch_size = 100
    >>> end_idx = 0
    >>> new_idx, X_batch, fX_batch, dfXdX_batch = slice_data(X, fX, dfXdX, batch_size, end_idx)
    >>> print(new_idx, X_batch.shape, fX_batch.shape, dfXdX_batch.shape)
    """
    return (
        end_idx + batch_size,
        jittable_slice(X, end_idx, batch_size),
        jittable_slice(fX, end_idx, batch_size),
        jittable_slice(dfXdX, end_idx, batch_size),
    )


def save_to_pickle(file_path: Path, data: Any) -> None:
    """
    Saves a Python object to a file using pickle serialization, ensuring the file
    has an appropriate extension for pickle files.

    Parameters
    ----------
    file_path : Path
        The path where the pickle file should be saved. If the file extension is not
        one of the recognized pickle formats, '.pkl' will be appended.
    data : Any
        The Python object to be serialized and saved. This can be any object that
        pickle can serialize.

    Returns
    -------
    None

    Notes
    -----
    The function checks the file extension and appends '.pkl' if the existing extension
    is not recognized as a pickle format. It ensures that the directory for the file path
    exists before saving. The data is serialized using pickle's highest protocol available
    for efficiency.

    This function handles disk I/O and may raise IOError if there are issues writing to
    the file. It uses 'wb+' mode to open the file, which will truncate the file to zero
    length if it already exists or create a new file if it does not exist.

    Example
    -------
    >>> from pathlib import Path
    >>> my_data = {'key': 'value'}
    >>> file_path = Path("/path/to/save/my_data.pkl")
    >>> save_to_pickle(file_path, my_data)
    """
    ext = file_path.suffix
    # Check if the file extension is one of the recognized pickle formats
    if ext not in {".pkl", ".pickle", ".pk", ".pck", ".pcl", ".p"}:
        ext = ".pkl"
    else:
        ext = ""

    # Ensure the directory exists
    makedirs(file_path.parents[0], exist_ok=True)

    # Open the file with the correct extension and save the data
    with open(file_path.with_suffix(file_path.suffix + ext), "wb+") as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path: Path) -> Any:
    """
    Loads and deserializes data from a pickle file specified by the given path.

    Parameters
    ----------
    file_path : Path
        The path to the pickle file. The file extension should be one of the
        following to avoid adding an additional '.pkl' extension: '.pkl', '.pickle',
        '.pk', '.pck', '.pcl', '.p'.

    Returns
    -------
    Any
        The deserialized object from the pickle file.

    Raises
    ------
    IOError
        If there is an issue reading the file.
    ValueError
        If the file extension is not recognized as a valid pickle format, although
        the function attempts to correct common extension errors by adding '.pkl'.

    Notes
    -----
    This function is designed to handle various common file extensions for serialized
    data using Python's pickle module. If the file extension is one of the recognized
    pickle formats, it loads the file directly. If not, it appends '.pkl' to the
    existing extension and attempts to load the file.

    Example
    -------
    >>> from pathlib import Path
    >>> file_path = Path("/path/to/data.pkl")
    >>> data = load_pickle(file_path)
    >>> print(type(data))
    """
    ext = file_path.suffix
    # Ensure that the file extension is recognized as a pickle format
    if ext not in {".pkl", ".pickle", ".pk", ".pck", ".pcl", ".p"}:
        ext = ".pkl"
    else:
        ext = ""

    # Open the file with the correct extension
    with open(file_path.with_suffix(file_path.suffix + ext), "rb") as file:
        deserialized = pickle.load(file)

    return deserialized


def load_1D_jax_array_direct_to_gpu(file_path: str, dtype=None) -> jax.Array:
    """
    Loads a binary file from disk directly into a 1D JAX array on the GPU.

    Parameters
    ----------
    file_path : str
        The path to the binary file containing the data to be loaded. Assumes that the
        data in the file is stored as `float64` and the file contains only 1D data.

    Returns
    -------
    jax.Array
        A 1D JAX array containing the data loaded from the specified file, converted
        to `float64` data type, and stored directly on the GPU.

    Notes
    -----
    This function uses NumPy's `fromfile` function to read raw binary data from a file,
    with an offset of 128 bytes to skip any headers or metadata present in the file format.
    The data is then cast to a JAX array to take advantage of JAX's GPU acceleration
    capabilities.

    The `like` parameter in `np.fromfile` uses a workaround through RAPIDAI's kvikio,
    `LikeWrapper` for compatibility with JAX array handling. If GPUDirectStorage from NVIDIA
    is available, it is used.

    Example
    -------
    >>> file_path = "/path/to/data.bin"
    >>> data_array = load_1D_jax_array_direct_to_gpu(file_path)
    >>> print(data_array.shape)
    """
    if dtype is not None:
        return jnp.asarray(
            np.fromfile(file_path, dtype=dtype, like=LikeWrapper(np.empty(())), offset=128)
        )
    else:
        return jnp.asarray(
            np.fromfile(file_path, like=LikeWrapper(np.empty(())), offset=128)
        )

def __load_shaped_jax_array_direct_to_gpu(
    file_path: str, shape: Tuple[int, ...]
) -> jax.Array:
    """
    Loads a binary file from disk directly into a JAX array on the GPU, with a specified shape.

    Parameters
    ----------
    file_path : str
        The full path to the binary file containing the data to be loaded.
    shape : Tuple[int, ...]
        The shape to which the loaded data should be reshaped. This tuple defines
        the dimensions of the resulting JAX array.

    Returns
    -------
    jax.Array
        A JAX array containing the data loaded from the specified file, reshaped
        to the given shape, and stored directly on the GPU.

    Notes
    -----
    This function reads raw binary data from a file using NumPy's `fromfile` method.
    The data is assumed to be of type `float64`. An offset of 128 bytes is used when
    reading the file, which might be required to skip headers or metadata present in
    the file format. The data is then converted to a JAX array, leveraging JAX's ability
    to utilize GPU resources for data storage and computation.

    This function is designed to be used internally and handles specific data loading
    tasks that are optimized for performance in a JAX environment.

    The `like` parameter in `np.fromfile` uses a workaround through RAPIDAI's kvikio,
    `LikeWrapper` for compatibility with JAX array handling.  If GPUDirectStorage from NVIDIA
    is available, it is used.

    Example
    -------
    >>> file_path = "/path/to/data.bin"
    >>> shape = (1000, 20)
    >>> data_array = __load_shaped_jax_array_direct_to_gpu(file_path, shape)
    """
    return jnp.asarray(
        np.fromfile(file_path, like=LikeWrapper(np.empty(())), offset=128).reshape(
            shape
        )
    )


def load_data_disk_direct_to_gpu_no_jacobians(
    data_config_dict: Dict[str, Any]
) -> Tuple[jax.Array, jax.Array]:
    """
    Loads input features and output features directly into GPU memory from disk,
    based on specified filenames and directory in the configuration dictionary.

    Parameters
    ----------
    data_config_dict : Dict[str, Any]
        A dictionary containing configuration settings for loading the data. Keys
        should include:
        - 'dir': Directory path where the data files are located.
        - 'N': The number of samples to load.
        - 'filenames': A tuple of filenames for the data arrays: X and fX.

    Returns
    -------
    Tuple[jax.Array, jax.Array]
        A tuple containing two JAX arrays loaded into GPU memory:
        - X: Input feature array.
        - fX: Output feature array.

    Notes
    -----
    This function is designed for cases where only the input features (X) and the output
    features (fX) are needed without the jacobians. It uses the utility function
    `__load_shaped_jax_array_direct_to_gpu` to read the data from disk and reshape it
    as specified, making it ready for computational use directly on GPU.

    Example
    -------
    >>> data_config = {
        "dir": "/path/to/data/",
        "N": 1000,
        "filenames": ("X.npy", "fX.npy")
    }
    >>> X, fX = load_data_disk_direct_to_gpu_no_jacobians(data_config)
    """
    data_dir = data_config_dict["dir"]
    N = data_config_dict["N"]
    X_filename, fX_filename = data_config_dict["filenames"][0:2]

    # Load data directly to GPU, reshape as required
    X = __load_shaped_jax_array_direct_to_gpu(Path(data_dir, X_filename), (N, -1))
    fX = __load_shaped_jax_array_direct_to_gpu(Path(data_dir, fX_filename), (N, -1))

    return X, fX


def load_data_disk_direct_to_gpu(
    data_config_dict: Dict[str, Any]
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Loads dataset arrays directly into GPU memory from specified filenames and
    directory in the configuration dictionary.

    Parameters
    ----------
    data_config_dict : Dict[str, Any]
        A dictionary containing configuration settings for loading the data. Keys
        should include:
        - 'dir': Directory path where the data files are located.
        - 'N': The number of samples to load.
        - 'filenames': A tuple of filenames for the data arrays: X, fX, and dfXdX.

    Returns
    -------
    Tuple[jax.Array, jax.Array, jax.Array]
        A tuple containing three JAX arrays loaded into GPU memory:
        - X: Input feature array.
        - fX: Output feature array.
        - dfXdX: Derivative of the output with respect to the input.

    Notes
    -----
    This function assumes that the data is stored in binary format compatible with JAX
    array loading capabilities. The function `__load_shaped_jax_array_direct_to_gpu` is
    used to read the data from disk and shape it according to the provided dimensions.
    This includes reshaping dfXdX to match the dimensions given the number of samples and
    the dimensions of X, specifically designed for handling derivative data.

    Example
    -------
    >>> data_config = {
        "dir": "/path/to/data/",
        "N": 1000,
        "filenames": ("X.bin", "fX.bin", "dfXdX.bin")
    }
    >>> X, fX, dfXdX = load_data_disk_direct_to_gpu(data_config)
    """
    data_dir = data_config_dict["dir"]
    N = data_config_dict["N"]
    X_filename, fX_filename, dfXdX_filename = data_config_dict["filenames"]

    # Load data directly to GPU, reshape as required
    X = __load_shaped_jax_array_direct_to_gpu(Path(data_dir, X_filename), (N, -1))
    fX = __load_shaped_jax_array_direct_to_gpu(Path(data_dir, fX_filename), (N, -1))
    dfXdX = __load_shaped_jax_array_direct_to_gpu(
        Path(data_dir, dfXdX_filename), (N, X.shape[1], -1)
    )

    return X, fX, dfXdX


def split_training_testing_data(
    data: Tuple[jax.Array, ...],
    data_config_dict: Dict[str, int],
    calculate_norms: bool = False,
) -> Tuple[List[jax.Array], List[jax.Array]]:
    """
    Splits the given dataset into training and testing sets according to specified
    sizes, and optionally calculates norms for the outputs and their jacobians.

    Parameters
    ----------
    data : Tuple[jax.Array, ...]
        A tuple containing the datasets to be split. The first array in the tuple
        is typically the input features (X), followed by output values (Y) and
        optionally jacobians (dYdX).
    data_config_dict : Dict[str, int]
        A dictionary containing the configuration for the data split, specifically
        the sizes of the training and testing datasets under keys 'nTrain'
        and 'nTest'.
    calculate_norms : bool, optional
        If True, calculates and appends the norms of output values and jacobians
        to the datasets. This is useful for error normalization and other relative
        calculations. Defaults to False.

    Returns
    -------
    Tuple[List[jax.Array], List[jax.Array]]
        A tuple of two lists: the first list contains arrays corresponding to the
        training data, and the second list contains arrays corresponding to the
        testing data. Each list contains arrays in the same order as provided in
        the input tuple `data`.

    Raises
    ------
    AssertionError
        If the number of data points in any array does not match or if the total
        number of data points is insufficient for the split as per the provided
        configuration.

    Notes
    -----
    The function optionally computes the squared L2 norms for the output arrays
    and jacobians if `calculate_norms` is set to True. This additional computation
    involves vectorized mapping over the arrays using JAX's `vmap` and `jit` for
    efficiency.

    Examples
    --------
    >>> X = jnp.array([...])
    >>> Y = jnp.array([...])
    >>> dYdX = jnp.array([...]) if jacobians are provided
    >>> data_config = {'nTrain': 100, 'nTest': 50}
    >>> train_data, test_data = split_training_testing_data((X, Y, dYdX), data_config)
    >>> # If calculate_norms is True, additional arrays for norms will be included.
    """
    n_test = data_config_dict["nTest"]
    n_train = data_config_dict["nTrain"]
    n_train_test = n_train + n_test

    if calculate_norms:
        print("Computing data norms for relative error calculations")
        norms = [
            vmap(jit(lambda x: jnp.linalg.norm(x) ** 2))(array) for array in data[1:]
        ]
        data = tuple(list(data) + norms)

    n_data, dM = data[0].shape
    assert all(
        (array.shape[0] == n_data for array in data)
    ), "All data arrays must have the same number of elements."
    assert (
        n_data >= n_train_test
    ), "Total data size must be at least the sum of training and testing sizes."

    return (
        tuple(array[:n_train] for array in data),
        tuple(array[n_train:n_train_test] for array in data),
    )


def split_training_testing_data_flat(
    data: Tuple[jax.Array], data_config_dict: Dict, calculate_norms: bool = False
) -> Tuple[Tuple[jax.Array], Tuple[jax.Array]]:
    """
    Splits the provided data into training and testing sets according to specified sizes,
    optionally computing norms for relative error calculations if needed.

    Parameters
    ----------
    data : Tuple[jax.Array]
        A tuple of JAX arrays containing the data to be split. Typically, this would include
        arrays for input features X, output values Y, and possibly jacobians dYdX.
    data_config_dict : Dict
        Configuration dictionary specifying sizes for training and testing data sets
        with keys 'nTrain' and 'nTest'.
    calculate_norms : bool, optional
        If True, compute norms for the output values and jacobians. This is useful for
        normalization purposes in error calculations. Defaults to False.

    Returns
    -------
    Tuple[Tuple[jax.Array], Tuple[jax.Array]]
        A tuple containing two tuples, each with arrays split into training and testing data.
        Training data arrays are first, followed by testing data arrays.

    Notes
    -----
    This function supports data where jacobians are provided alongside output values.
    If `calculate_norms` is True, it will modify the output values by concatenating them
    with a flattened version of the jacobians, then calculate norms across these modified outputs.
    Each array in `data` must have the same number of data points (i.e., same size in the first dimension).

    Examples
    --------
    >>> X = jnp.array([...])
    >>> Y = jnp.array([...])
    >>> dYdX = jnp.array([...])
    >>> data_config = {'nTrain': 100, 'nTest': 50}
    >>> train_data, test_data = split_training_testing_data_flat((X, Y, dYdX), data_config)
    """
    n_test = data_config_dict["nTest"]
    n_train = data_config_dict["nTrain"]
    n_train_test = n_train + n_test
    if calculate_norms:
        print("Computing data norms for relative error calculations")
        Y_dYdX = jnp.concatenate([data[1], vmap(lambda x: x.ravel())(data[2])], axis=1)
        if data_config_dict['jacobian']: #(X, Y, dYdX) -> (X, Y_dYdX) + norms
            data = [data[0], Y_dYdX] + [
            vmap(jit(lambda x: jnp.linalg.norm(x) ** 2))(array) for array in data[1:]
        ]
        else: # (X, Y, Y_dYdX) + norms
            data = [data[0], data[1], Y_dYdX] + [
            vmap(jit(lambda x: jnp.linalg.norm(x) ** 2))(array) for array in data[1:]
        ]
        for datum in data:
            print ("shapes", datum.shape)

    n_data, dM = data[0].shape
    print(f"Total data: {n_data}, Training data: {n_train}, Testing data: {n_test}")

    assert all(
        (array.shape[0] == n_data for array in data)
    ), "All data arrays must have the same number of elements."
    assert (
        n_data >= n_train_test
    ), "Total data size must be at least the sum of training and testing sizes."

    return (
        tuple(array[:n_train] for array in data),
        tuple(array[n_train:n_train_test] for array in data),
    )
