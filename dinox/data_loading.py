from pathlib import Path
from typing import Dict, Iterable, Tuple

import hickle
import jax.numpy as jnp
from jax import Array as jax_Array
from jax import default_device as jax_default_device
from jax import devices as jax_devices
from jax import device_put
from jax.tree_util import tree_map
from jaxtyping import PyTree
from numpy.typing import ArrayLike
import numpy as np
from .losses import cpu_vectorized_squared_norm, vectorized_squared_norm


def dump_pytree_to_disk(pytree: PyTree, path: Path) -> None:
    hickle.dump(tree_map(np.asarray, pytree), path, mode="w")


def load_pytree_from_disk(path: Path) -> PyTree:
    return tree_map(jnp.array, hickle.load(path))


def load_pytree_as_dict_of_np_arrays_from_disk(
    path: Path,
) -> Dict[str, ArrayLike]:
    return hickle.load(path)


def device_put_pytree(pytree: PyTree) -> PyTree:
    return tree_map(device_put, pytree)


def device_get_pytree(pytree: PyTree) -> PyTree:
    return tree_map(device_put, pytree)


def add_squared_norms_of_each_entry(
    data_dict: Dict[str, jax_Array],
) -> Dict[str, jax_Array]:
    device = data_dict.popitem()[1].device()
    if device in jax_devices("cpu"):
        norm_func = cpu_vectorized_squared_norm
    else:
        norm_func = vectorized_squared_norm
    with jax_default_device(device):
        data_dict |= {f"{data_key}_norms": norm_func(data) for data_key, data in data_dict.items() if data_key != "X"}
    return data_dict


def load_encoded_training_validation_and_testing_data(
    REDUCED_DIMS: Tuple[int, int],
    data_keys: Iterable[str],
    renaming_dict: Dict[str, str],
    N_TRAIN: int,
    N_VAL: int = 2500,
    N_TEST: int = 25_000,
) -> Dict[str, Dict[str, ArrayLike]]:
    """
    Loads training and testing data from .npy files and computes squared L2 norms for testing data.

    Parameters:
        REDUCED_DIMS (Tuple[int, int]): Reduced dimensions.
        data_keys (Iterable[str]): Data keys to load.
        N_TRAIN (int): Number of training samples.
        N_TEST (int, optional): Number of testing samples (default is 25_000; truncated if lower).

    Returns:
        Dict[str, Dict[str, np.ndarray]]:
            - "training data": Maps each key to its training array.
            - "testing data": Maps each key to its testing array and includes computed norms (keyed as "{data_key}_norms").
    """
    # Load training data onto GPU
    train_data = {
        renaming_dict[data_key]: jnp.load(f"{REDUCED_DIMS}_training_{data_key}_{N_TRAIN}.npy")[:40]  # training_
        for data_key in data_keys
    }
    for key, val in train_data.items():
        print(key, val.shape)
    # Load validation data onto GPU; truncate if N_VAL is less than 2500
    if N_VAL > 0:
        if N_VAL < 2500:
            val_data = {
                renaming_dict[data_key]: jnp.load(f"validation_{REDUCED_DIMS}_{data_key}_{N_VAL}.npy")[:N_VAL]
                for data_key in data_keys
            }
        else:
            val_data = {
                renaming_dict[data_key]: jnp.load(f"{REDUCED_DIMS}_training_{data_key}_50.npy")
                for data_key in data_keys
            }

    # Load testing data onto CPU; truncate if N_TEST is less than 25000
    with jax_default_device(jax_devices("cpu")[0]):
        if N_TEST < 25_000:
            cpu_test_data = {
                renaming_dict[data_key]: jnp.load(f"testing_{REDUCED_DIMS}_{data_key}_25000.npy")[:N_TEST]
                for data_key in data_keys
            }
        else:
            cpu_test_data = {
                renaming_dict[data_key]: jnp.load(f"{REDUCED_DIMS}_training_{data_key}_{25000}.npy")  # _50000.npy')
                for data_key in data_keys
            }

    print("Computing data norms for relative testing error calculations")

    # Compute squared L2 or Frobenius norms for each validation/testing data array and add to dictionaries.
    train_val_test = {
        "train": train_data,
        "test": add_squared_norms_of_each_entry(cpu_test_data),
    }
    if N_VAL > 0:
        train_val_test["val"] = add_squared_norms_of_each_entry(val_data)
    return train_val_test


def check_batch_size_validity(*, data_iterable: Iterable[ArrayLike], batch_size: int) -> int:
    n_batches_list = []
    n_data = next(iter(data_iterable)).shape[0]
    for data_tuple in data_iterable:
        n_data_i = data_tuple.shape[0]
        assert n_data_i == n_data, "We require the same number of data for inputs and all outputs"
        n_batches, remainder = divmod(n_data_i, batch_size)
        if remainder != 0:
            raise ValueError("Adjust `batch_size` to evenly divide training and testing data.")
        n_batches_list.append(n_batches)
    assert n_batches_list[:-1] == n_batches_list[1:], "The data does not all have the same number of elements!"
    return n_batches
