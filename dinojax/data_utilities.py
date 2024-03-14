import equinox as eqx
import jax
from jax.lax import dynamic_slice_in_dim
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import numpy as np
from kvikio.numpy import LikeWrapper
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple 

def create_array_permuter(N) -> Callable:
	indices = jnp.arange(N)
	@jax.jit
	def permute_arrays(
					X: jax.Array, 
					Y: jax.Array,
					dYdX: jax.Array,
					Y_norms: jax.Array,
					dYdX_norms: jax.Array,
					key:jr.PRNGKey
					) -> Tuple[jr.PRNGKey, Tuple[jax.Array]]:
		(key, subkey) = jr.split(key)
		perm = jr.permutation(subkey, indices)
		return key, (X[perm], Y[perm], dYdX[perm], Y_norms[perm], dYdX_norms[perm])
	return permute_arrays

@eqx.filter_jit
def slice_data(X: jax.Array, Y: jax.Array, dYdX: jax.Array, batch_size: int, end_idx:int) -> Tuple[jax.Array,jax.Array,jax.Array]:
	return (end_idx + batch_size, 
		dynamic_slice_in_dim(X, end_idx, batch_size),
		dynamic_slice_in_dim(Y, end_idx, batch_size), 
		dynamic_slice_in_dim(dYdX, end_idx, batch_size))

def save_to_pickle(file_path: Path, data: Any) -> None:
	# Involves Disk I/O
	ext = file_path.suffix
	if not ext:
		ext = '.pkl'
	makedirs(file_path.parents[0], exist_ok = True)
	with open(Path.joinpath(file_path, ext), 'wb+') as file:
		pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

def load_data_disk_direct_to_gpu(data_config_dict: Dict) -> Tuple[jax.Array,
																  jax.Array,
																  jax.Array]:
	data_dir = data_config_dict['data_dir']
	N = data_config_dict['N']
	X_filename, Y_filename, dYdX_filename = data_config_dict['data_filenames']

	X = jnp.asarray(np.fromfile(data_dir+X_filename, like=LikeWrapper(np.empty(())),offset=128),dtype=np.float64).reshape((N,-1))
	Y = jnp.asarray(np.fromfile(data_dir+Y_filename, like=LikeWrapper(np.empty(())),offset=128)).reshape((N,-1))
	dYdX  = jnp.asarray(np.fromfile(data_dir+dYdX_filename, like=LikeWrapper(np.empty(())),offset=128)).reshape((N,X.shape[1],-1))
	return X, Y, dYdX

def	split_training_testing_data(data: Tuple[jax.Array],
								data_config_dict: Dict
								) -> Tuple[Tuple[jax.Array],
										   Tuple[jax.Array]]:
	n_test  = data_config_dict['test_data_size']
	n_train = data_config_dict['train_data_size']
	n_train_test = n_train + n_test
	#data = X,Y,dYdX
	# Y_norms, dYdX_norms = [vmap(jnp.linalg.norm)(array) for array in data[1:]] 
	data = list(data) + [vmap(jnp.linalg.norm)(array) for array in data[1:]] 
	n_data, dM = data[0].shape
	assert all((array.shape[0] == n_data for array in data))
	assert n_data >= n_train_test
	return ([array[:n_train] for array in data], 
			[array[n_train:n_train_test] for array in data])


