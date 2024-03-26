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

from os import makedirs
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple

import cupy as cp
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import vmap
from jax.lax import dynamic_slice_in_dim as dslice
from kvikio.numpy import LikeWrapper


def sub_dict(*, super_dict: Dict, keys: Iterable):
    # No side effects
    return {k: super_dict[k] for k in keys}


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


def create_array_permuter(N, cp_random_seed=None) -> Callable:
    # Has side effects! Stateful cp_random_seed

    indices = cp.arange(N)
    if cp_random_seed is not None:
        cp.random.seed(cp_random_seed)

    def permute_arrays(
        X: jax.Array,
        Y: jax.Array,
        dYdX: jax.Array,
        Y_norms: jax.Array,
        dYdX_norms: jax.Array,
    ) -> Tuple[jax.Array]:
        perm = cp.random.permutation(indices)

        return (
            jax.dlpack.from_dlpack(cp.from_dlpack(jax.dlpack.to_dlpack(X))[perm]),
            jax.dlpack.from_dlpack(cp.from_dlpack(jax.dlpack.to_dlpack(Y))[perm]),
            jax.dlpack.from_dlpack(cp.from_dlpack(jax.dlpack.to_dlpack(dYdX))[perm]),
            jax.dlpack.from_dlpack(cp.from_dlpack(jax.dlpack.to_dlpack(Y_norms))[perm]),
            jax.dlpack.from_dlpack(
                cp.from_dlpack(jax.dlpack.to_dlpack(dYdX_norms))[perm]
            ),
        )

    return permute_arrays


@eqx.filter_jit
def slice_data(
    X: jax.Array, Y: jax.Array, dYdX: jax.Array, batch_size: int, end_idx: int
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    # No side effects
    return (
        end_idx + batch_size,
        dslice(X, end_idx, batch_size),
        dslice(Y, end_idx, batch_size),
        dslice(dYdX, end_idx, batch_size),
    )


def save_to_pickle(file_path: Path, data: Any) -> None:
    # Involves Disk I/O
    ext = file_path.suffix
    if not ext:
        ext = ".pkl"
    makedirs(file_path.parents[0], exist_ok=True)
    with open(Path.joinpath(file_path, ext), "wb+") as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

def load_pickle(file_path: Path, data: Any) -> None:
    # Involves Disk I/O
    ext = file_path.suffix
    if not ext:
        ext = ".pkl"
    with open(Path.joinpath(file_path, ext), "rb") as file:
        deserialized =  pickle.load(data, file, pickle.HIGHEST_PROTOCOL)
    return deserialized

def __load_jax_array_with_shape_direct_to_gpu(file_path, shape):
    # assumes float64
    return jnp.asarray(
        np.fromfile(file_path, like=LikeWrapper(np.empty(())), offset=128),
        dtype=np.float64,
    ).reshape(shape)


def load_data_disk_direct_to_gpu(
    data_config_dict: Dict,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    data_dir = data_config_dict["data_dir"]
    N = data_config_dict["N"]
    X_filename, Y_filename, dYdX_filename = data_config_dict["data_filenames"]
    # print(data_dir, X_filename, Y_filename, dYdX_filename)
    # print(N,mq_data['m_data'].shape,  mq_data['q_data'].shape,  np.load(cli_args['data_dir']+'JstarPhi_data.npz')['JstarPhi_data'].shape)	5000 (5000, 1681) (5000, 25) (5000, 1681, 25)
    X = __load_jax_array_with_shape_direct_to_gpu(data_dir + X_filename, (N, -1))
    Y = __load_jax_array_with_shape_direct_to_gpu(data_dir + Y_filename, (N, -1))
    # print(X.shape, Y.shape)
    dYdX = __load_jax_array_with_shape_direct_to_gpu(
        data_dir + dYdX_filename, (N, X.shape[1], -1)
    )
    return X, Y, dYdX


def split_training_testing_data(
    data: Tuple[jax.Array], data_config_dict: Dict, calculate_norms: bool = False
) -> Tuple[Tuple[jax.Array], Tuple[jax.Array]]:
    # No side effects
    n_test = data_config_dict["test_data_size"]
    n_train = data_config_dict["train_data_size"]
    n_train_test = n_train + n_test
    if calculate_norms:
        # data = X,Y,dYdX
        # Y_norms, dYdX_norms = [vmap(jnp.linalg.norm)(array) for array in data[1:]]
        data = list(data) + [vmap(jnp.linalg.norm)(array) for array in data[1:]]
    n_data, dM = data[0].shape
    assert all((array.shape[0] == n_data for array in data))
    assert n_data >= n_train_test
    return (
        [array[:n_train] for array in data],
        [array[n_train:n_train_test] for array in data],
    )
