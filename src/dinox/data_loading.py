from pathlib import Path
from typing import Dict, Iterable, Tuple

import hickle
import jax.random as jr
from jax import default_device as jax_default_device
from jax import device_get, device_put
from jax import devices as jax_devices
from jax.tree_util import tree_map
from jax.typing import ArrayLike as JAXArrayLike
from jaxtyping import PyTree
from numpy.typing import ArrayLike


def dump_pytree_to_disk(pytree: PyTree, path: Path) -> None:
    import numpy as np

    hickle.dump(tree_map(np.asarray, pytree), path, mode="w")


def load_pytree_from_disk(path: Path) -> PyTree:
    import jax.numpy as jnp

    return tree_map(jnp.array, hickle.load(path))


def load_pytree_as_dict_of_np_arrays_from_disk(
    path: Path,
) -> Dict[str, ArrayLike]:
    return hickle.load(path)


def device_put_pytree(pytree: PyTree) -> PyTree:
    return tree_map(device_put, pytree)


def device_get_pytree(pytree: PyTree) -> PyTree:
    return tree_map(device_get, pytree)


def add_squared_norms_of_each_entry(
    data_dict: Dict[str, JAXArrayLike],
) -> Dict[str, JAXArrayLike]:
    device = next(iter(data_dict[list(data_dict.keys())[0]].devices()))
    if device in jax_devices("cpu"):
        from .losses import __cpu_vectorized_squared_norm

        norm_func = __cpu_vectorized_squared_norm
    else:
        from .losses import __vectorized_squared_norm

        norm_func = __vectorized_squared_norm
    for data_key in list(data_dict.keys()):
        if data_key != "X":
            data_dict[f"{data_key}_norms"] = norm_func(data_dict[data_key])
    return data_dict


def load_encoded_training_validation_and_testing_data(
    data_path: Path,
    REDUCED_DIMS: Tuple[int, int],
    training_data_keys: Iterable[str],
    N_TRAIN: int,
    renaming_dict: Dict[str, str] = None,
    test_data_keys: Iterable[str] = ("encoded_inputs", "encoded_outputs", "encoded_Jacobians"),
    N_VAL: int = 2500,
    N_TEST: int = 10_000,  # 25_000,
) -> Dict[str, Dict[str, ArrayLike]]:
    import jax.numpy as jnp

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
    if renaming_dict is None:
        renaming_dict = {data_key: data_key for data_key in training_data_keys}

    # Load training data onto GPU-- useful only if training data saved to "training/f"{REDUCED_DIMS}_{data_key}_{N_TRAIN}.npy"
    train_data = {
        renaming_dict[data_key]: jnp.load(Path(data_path, "training", f"{REDUCED_DIMS}_{data_key}_{N_TRAIN}.npy"))
        for data_key in training_data_keys
    }

    with jax_default_device(jax_devices("cpu")[0]):
        cpu_train_data = {
            renaming_dict[data_key]: jnp.load(Path(data_path, "training", f"{REDUCED_DIMS}_{data_key}_{N_TRAIN}.npy"))
            for data_key in training_data_keys
        }

    # print("Taking stock of training data:")
    # for key, val in train_data.items():
    #     print(f"\t{key}, shape: {val.shape}")
    # Load validation data onto GPU; truncate if N_VAL is less than 2500
    if N_VAL > 0:
        if N_VAL < 2500:
            val_data = {
                renaming_dict[data_key]: jnp.load(Path(data_path, "validation", f"{REDUCED_DIMS}_{data_key}.npy"))[
                    :N_VAL
                ]
                for data_key in training_data_keys
            }
        else:
            val_data = {
                renaming_dict[data_key]: jnp.load(Path(data_path, "validation", f"{REDUCED_DIMS}_{data_key}.npy"))
                for data_key in training_data_keys
            }

    # Load testing data onto CPU; truncate if N_TEST is less than 25000
    # by default includes encoded_Jacobians, since we want to test the accuracy of Jacobians even when training without them
    with jax_default_device(jax_devices("cpu")[0]):
        if N_TEST < 10_000:
            cpu_test_data = {
                renaming_dict[data_key]: jnp.load(Path(data_path, "testing", f"{REDUCED_DIMS}_{data_key}.npy"))[:N_TEST]
                for data_key in test_data_keys
            }
        else:
            cpu_test_data = {
                renaming_dict[data_key]: jnp.load(Path(data_path, "testing", f"{REDUCED_DIMS}_{data_key}.npy"))
                for data_key in test_data_keys
            }

    # Compute squared L2 or Frobenius norms for each validation/testing data array and add to dictionaries.
    train_val_test = {
        "train": train_data,
        "train_cpu": add_squared_norms_of_each_entry(cpu_train_data),
        "test": add_squared_norms_of_each_entry(cpu_test_data),
    }
    if N_VAL > 0:
        train_val_test["val"] = add_squared_norms_of_each_entry(val_data)
    return train_val_test


def load_encoded_data_and_add_noise(
    data_path: Path, reduced_dims: Tuple[int, int], n_train: int, observation_noise_key: jr.PRNGKey
):
    training_data_keys = ("encoded_inputs", "encoded_outputs")
    test_data_keys = ("encoded_inputs", "encoded_outputs", "encoded_Jacobians")
    renaming_dict = {"encoded_inputs": "X", "encoded_outputs": "fX", "encoded_Jacobians": "dfXdX"}
    train_val_test = load_encoded_training_validation_and_testing_data(
        data_path=data_path,
        REDUCED_DIMS=reduced_dims,
        training_data_keys=training_data_keys,
        test_data_keys=test_data_keys,
        renaming_dict=renaming_dict,
        N_TRAIN=n_train,
    )
    TRAINING_DATA, VALIDATION_DATA, CPU_TEST_DATA = (
        train_val_test["train"],
        train_val_test["val"],
        train_val_test["test"],
    )

    # Vanilla amortized estimation trains with non-degenerate X,Y distribution data.
    # In our case, we need to add noise from our noise model: Y = f(X) + n, n\sim N(0, C).
    # Since we actually train on !encoded! data, with whitening encoding transformation y_r = E Y = Ef(x) + En, where E D = I_r, D = C E^T ,
    # we arrive at y_r \sim N(Ef(x), E C E.T) = N(Ef(x), I_r). Hence, we add white noise
    key_train, key_val = jr.split(observation_noise_key)
    TRAINING_DATA["Y"] = TRAINING_DATA["fX"] + jr.normal(key_train, TRAINING_DATA["fX"].shape)
    VALIDATION_DATA["Y"] = VALIDATION_DATA["fX"] + jr.normal(key_val, VALIDATION_DATA["fX"].shape)
    del TRAINING_DATA["fX"], VALIDATION_DATA["fX"]

    return TRAINING_DATA, VALIDATION_DATA, CPU_TEST_DATA, renaming_dict


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
