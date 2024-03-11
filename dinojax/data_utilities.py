import equinox as eqx
import jax
import jax.random as jr

from dataclasses import dataclass
from jax_dataclasses import pytree_dataclass
from typing import Dict, Iterable, Union, Tuple, TYPE_CHECKING # TypeAlias,

@pytree_dataclass
class TrainingDataClass():
	X: jax.Array
	Y: jax.Array

@pytree_dataclass
class JacTrainingDataclass():
	X: jax.Array
	Y: jax.Array
	dYdX: jax.Array

@pytree_dataclass
class NormsAugmentedJacTrainingDataclass():
	X: jax.Array
	Y: jax.Array
	dYdX: jax.Array
	Y_norms: jax.Array
	dYdX_norms: jax.Array

# if TYPE_CHECKING:
DinoTrainingDataclass = \
	Union[JacTrainingDataclass, 
			TrainingDataClass,
			NormsAugmentedJacTrainingDataclass] #: TypeAlias , python 3.10

#can we jit this???
def permute_arrays(arrays: Iterable[jax.Array],
				   *,
				   key:jr.PRNGKey
				   ) -> Tuple[jr.PRNGKey, Iterable[jax.Array]]:
	dataset_size = arrays[0].shape[0]
	perm = jr.permutation(key, jnp.arange(dataset_size))
	(key,) = jr.split(key, 1)
	permuted_arrays = [array[perm] for array in arrays]
	return key, permuted_arrays

@eqx.filter_jit
def slice_data(X: jax.Array, Y: jax.Array, dYdX: jax.Array, batch_size: int, end_idx:int) -> Tuple[jax.Array,jax.Array,jax.Array]:
	return (end_idx + batch_size, 
		dynamic_slice_in_dim(X, end_idx, batch_size),
		dynamic_slice_in_dim(Y, end_idx, batch_size), 
		dynamic_slice_in_dim(dYdX, end_idx, batch_size))

def load_data_from_disk(data_config_dict):
	data_dir = data_config_dict['data_dir']
	X_filename, Y_filename, dYdX_filename = data_config_dict['data_file_names']

	# import nvidia.dali.fn as fn
	# import nvidia.dali.types as types
	# from nvidia.dali.pipeline import Pipeline
	# from nvidia.dali import pipeline_def

	# files = sorted([f for f in os.listdir(args.data_dir) if ".npy" in f])

	# @pipeline_def(batch_size=1, num_threads=1, device_id=0)
	# def data_pipeline(filename):
	#     return fn.readers.numpy(device="gpu", file_root=args.data_dir, file_filter=f"*{filename}.npy", use_o_direct=True)

	# data = []
	from kvikio.numpy import LikeWrapper

	import jax.dlpack as jdl
	# for file_name in ["m_data", "q_data", "J_data"]: #, "q_data_norms", "J_data_norms"]:
		# data.append_idx(fn.readers.numpy(device="gpu", file_root=args.data_dir, file_filter=f"*{file_name}.npy", use_o_direct=True))
		# pipe = data_pipeline(file_name)
		# with pipe:
		# 	pipe.build()
		# 	data.append_idx(jdl.from_dlpack(pipe.run()[0].as_tensor()._expose_dlpack_capsule())[0])

	#convert to jnp arrays: 
	#take the numbers and use them for reshaping

	X = jnp.asarray(np.fromfile(data_dir+X_filename, like=LikeWrapper(np.empty(())),offset=128),dtype=np.float64).reshape((1000,400))
	Y = jnp.asarray(np.fromfile(data_dir+Y_filename, like=LikeWrapper(np.empty(())),offset=128)).reshape((1000,50))
	dYdX  = jnp.asarray(np.fromfile(data_dir+dYdX_filename, like=LikeWrapper(np.empty(())),offset=128)).reshape((1000,50,400))
	# print(X.shape, Y.shape, dYdX.shape)
	# from jax import device_put

	# m_data = device_put(jnp.array(data['m_data']))
	# u_data = device_put(jnp.array(data['q_data']))
	# J_data = device_put(jnp.array(data['J_data']))
	# 
	return JacTrainingDataclass(
		X=X,
		Y=Y,
		dYdX=dYdX)

def	split_training_testing_data(data: DinoTrainingDataclass,
								data_config_dict: Dict
								) -> Tuple[DinoTrainingDataclass,
										   DinoTrainingDataclass]:
	n_test  = data_config_dict['test_data_size']
	n_train = data_config_dict['train_data_size']

	Y_norms = vmap(jax.numpy.linalg.norm)(data.Y)
	dYdX_norms = vmap(jax.numpy.linalg.norm)(data.dYdX)

	n_data, dM = data.X.shape
	assert n_data == data.Y.shape[0] == \
		   data.dYdX.shape[0] == data.Y_norms.shape[0]== data.dYdX_norms.shape[0]
	assert n_data >= n_test + n_train

	training_data = NormsAugmentedJacTrainingDataclass(
		X=data.X[:n_train],
		Y=data.Y[:n_train],
		dYdX = data.dYdX[:n_train],
		Y_norms= Y_norms[:n_train],
		dYdX_norms = dYdX_norms[:n_train])
	#TODO, INITIALIZE NETWORK WITH RANDOM WEIGHTS, SAVE NETWORK (Serialize)
	n_train_test = n_train + n_test
	testing_data = NormsAugmentedJacTrainingDataclass(
		X=data.X[n_train:n_train_test],
		Y=data.Y[n_train:n_train_test],
		dYdX = data.dYdX[n_train:n_train_test],
		Y_norms= Y_norms[n_train:n_train_test],
		dYdX_norms = dYdX_norms[n_train:n_train_test])	
	return training_data, testing_data


