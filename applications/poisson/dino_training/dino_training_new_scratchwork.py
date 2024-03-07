import sys, os
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', '--batch_size', type = int, default = 20, help = 'gradient batch size')
parser.add_argument('-step_size', '--step_size', type=float, default=1e-3, help="What step size or 'learning rate'?")
parser.add_argument('-num_epochs', '--num_epochs', type=int, default=1000, help="How many epochs")
parser.add_argument('-visible_gpu', '--visible_gpu', type=int, default=0, help="Visible CUDA device")
parser.add_argument('-run_seed', '--run_seed', type=int, default=0, help="Seed for data shuffling / initialization")

parser.add_argument('-rb_choice', '--rb_choice', type=str, default='as', help="choose from [as, kle / pca, None]")
parser.add_argument('-rb_dir', '--rb_dir', type=str, default='../reduced_bases/', help="Where to load the reduced bases from")
parser.add_argument('-rb_rank', '--rb_rank', type=int, default=100, help="RB dim")
parser.add_argument('-depth', '--depth', type=int, default=5, help="RB dim")

parser.add_argument('-lr_schedule', '--lr_schedule', type=str, default='piecewise', help="Use LR Schedule or do not")
parser.add_argument('-weight_decay', '--weight_decay', type=float, default=0.0, help="Weight decay parameter")

parser.add_argument('-data_dir', '--data_dir', type=str, default='../data/pointwise/', help="What directory for all data to be split")


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.visible_gpu)

sys.path.append('../../../')

import time
import functools

import jax.numpy as jnp 
import numpy as np
import optax 

# from flax.training import train_state
from jax import jit
from jax.lax import dynamic_slice_in_dim

# import dinojax as dj


# Load the data


# with open(args.data_dir+'m_data.npy', 'wb') as f:
# 	print(data['m_data'].shape, data['m_data'].dtype)
# 	np.save(f, data['m_data'])
# with open(args.data_dir+'q_data.npy', 'wb') as f:
#     np.save(f, data['q_data'])
# with open(args.data_dir+'J_data.npy', 'wb') as f:
#     np.save(f, data['J_data'])
# with open(args.data_dir+'q_data_norms.npy', 'wb') as f:
#     np.save(f, vmap(jax.numpy.linalg.norm)(data['q_data']))
# with open(args.data_dir+'J_data_norms.npy', 'wb') as f:
#     np.save(f, vmap(jax.numpy.linalg.norm)(data['J_data']))



# import nvidia.dali.fn as fn
# import nvidia.dali.types as types
# from nvidia.dali.pipeline import Pipeline
# from nvidia.dali import pipeline_def

# files = sorted([f for f in os.listdir(args.data_dir) if ".npy" in f])

# @pipeline_def(batch_size=1, num_threads=1, device_id=0)
# def data_pipeline(filename):
#     return fn.readers.numpy(device="gpu", file_root=args.data_dir, file_filter=f"*{filename}.npy", use_o_direct=True)

data = []
from kvikio.numpy import LikeWrapper

import jax.dlpack as jdl
# for file_name in ["m_data", "q_data", "J_data"]: #, "q_data_norms", "J_data_norms"]:
	# data.append(fn.readers.numpy(device="gpu", file_root=args.data_dir, file_filter=f"*{file_name}.npy", use_o_direct=True))
	# pipe = data_pipeline(file_name)
	# with pipe:
	# 	pipe.build()
	# 	data.append(jdl.from_dlpack(pipe.run()[0].as_tensor()._expose_dlpack_capsule())[0])
m_data = jnp.asarray(np.fromfile(args.data_dir+"m_data.npy", like=LikeWrapper(np.empty(())),offset=128),dtype=np.float64).reshape((1000,400))
u_data = jnp.asarray(np.fromfile(args.data_dir+"q_data.npy", like=LikeWrapper(np.empty(())),offset=128)).reshape((1000,50))
J_data  = jnp.asarray(np.fromfile(args.data_dir+"J_data.npy", like=LikeWrapper(np.empty(())),offset=128)).reshape((1000,50,400))
# print(m_data.shape, u_data.shape, J_data.shape)
#, u_data_norms, J_data_norms python 
# from jax import device_put

# m_data = device_put(jnp.array(data['m_data']))
# u_data = device_put(jnp.array(data['q_data']))
# J_data = device_put(jnp.array(data['J_data']))
# u_data_norms = vmap(jax.numpy.linalg.norm)(u_data)
# J_data_norms = vmap(jax.numpy.linalg.norm)(J_data)
# print(J_data.shape, u_data.shape)

assert m_data.shape[0] == u_data.shape[0] == J_data.shape[0]

n_data, dM = m_data.shape
_, dU = u_data.shape
n_train = 800
n_test = n_data - n_train

m_train = m_data[:n_train]
u_train = u_data[:n_train]
J_train = J_data[:n_train]

#TODO, INITIALIZE NETWORK WITH RANDOM WEIGHTS, SAVE NETWORK (Serialize)
# u_norms_train = u_data_norms[:n_train]
# J_norms_train = J_data_norms[:n_train]

m_test = m_data[n_train:]
u_test = u_data[n_train:]
J_test = J_data[n_train:]
# u_norms_test = u_data_norms[n_train:]
# J_norms_test = J_data_norms[n_train:]

batch_size = args.batch_size 
num_epochs = args.num_epochs

n_train_batches, remainder = divmod(n_train, batch_size)
n_test_batches, remainder_test = divmod(n_test, batch_size)

if remainder != 0 :
	print(f"Warning, the number of training data ({n_train}) does not evenly devide into n_batches={n_train_batches}")
if remainder_test != 0 :
	print(f"Warning, the number of testing data ({n_test}) does not evenly devide into n_batches={n_test_batches}")
	#POSSIBLE FUTURE TODO: allow remainder > 0

train_steps_per_epoch = n_train_batches + int(bool(remainder))
num_train_steps = train_steps_per_epoch * num_epochs

if False:
	# LR scheduling does not seem to help in this case, but this is how you do it. 
	print('Using piecewise lr schedule')
	lr_schedule = optax.piecewise_constant_schedule(init_value=args.step_size,
						 boundaries_and_scales={int(num_train_steps*0.95):0.1,
												int(num_train_steps*0.975):0.1})
else:
	lr_schedule = args.step_size
# Set up the optimizer

optimizer = optax.adam(learning_rate=lr_schedule)

nn_widths = args.depth*[2*args.rb_rank]

import equinox as eqx 
import jax.random as jr
rng_key = jr.PRNGKey(0)

loader_key, model_key = jr.split(rng_key, 2)

nn_width = 2*args.rb_rank

nn = dj.GenericDenseFactory(layer_width = nn_width, depth=args.depth, input_size=dM, output_size=dU)

optimizer_state = optimizer.init(eqx.filter(nn, eqx.is_inexact_array))


# @jit
# def initialize(params_rng):
# 	init_rngs = {'params': params_rng}
# 	input_shape = (1, dM)
# 	variables = network.init(init_rngs, jnp.ones(input_shape, jnp.float32))
# 	return variables

# Initialize parameters of the model + optimizer.
from jax import vjp, vmap
def value_and_jacrev(f, xs):
    _, pullback =  vjp(f, xs[0])
    basis = jnp.eye(_.size, dtype=_.dtype)

    @jit
    def value_and_jacrev_x(x):
        y, pullback = vjp(f, x)
        jac = vmap(pullback)(basis)
        return y, jac[0] #
    return vmap(value_and_jacrev_x)(xs)

@eqx.filter_jit
def mean_h1_seminorm_loss(nn, input_X, actual_Y, actual_dYdX):
	predicted_Y, predicted_dYdX =  value_and_jacrev(nn, input_X)
	# predicted_Y = predicted_Y.squeeze()
	# predicted_dYdX = predicted_dYdX.squeeze()
	return jnp.mean(optax.l2_loss(predicted_Y.squeeze(), actual_Y)) + jnp.mean(optax.l2_loss(predicted_dYdX.squeeze(), actual_dYdX))*dM

grad_loss_fn = eqx.filter_grad(mean_h1_seminorm_loss)

@eqx.filter_jit
def take_step(	optimizer_state,
			  	nn,
				input_X,
				actual_Y,
				actual_dYdX
				):
	updates, optimizer_state = optimizer.update(
		grad_loss_fn(nn, 
					 input_X, actual_Y, actual_dYdX),
		optimizer_state)
	return optimizer_state, eqx.apply_updates(nn, updates)

@eqx.filter_jit
def mean_h1_seminorm_l2_errors_and_norms(nn, X, Y, dYdX):
	predicted_Y, predicted_dYdX =  value_and_jacrev(nn, dynamic_slice_in_dim(X, end, batch_size))
	# predicted_Y = predicted_Y.squeeze()
	# predicted_dYdX = predicted_dYdX.squeeze()
	# batch_se  = jnp.mean(optax.l2_loss(predicted_Y.squeeze(), Y),axis=1)
	# batch_sje = jnp.mean(optax.l2_loss(predicted_dYdX.squeeze(), dYdX)*dM,axis=(1,2))
	mse_i, msje_i = jnp.mean(optax.l2_loss(predicted_Y.squeeze(), dynamic_slice_in_dim(Y, end, batch_size)),axis=1), jnp.mean(optax.l2_loss(predicted_dYdX.squeeze(),
	dynamic_slice_in_dim(dYdX, end, batch_size))*dM,axis=(1,2))
	return  end + batch_size, one_over_n_batches*jnp.mean(mse_i),
	one_over_n_batches*jnp.mean(mse_i),
	one_over_n_batches*jnp.mean(mse_i), one_over_n_batches*jnp.mean(normalize_values(mse_i, dynamic_slice_in_dim(Y_L2_norms, end, batch_size))), one_over_n_batches*jnp.mean(normalize_values(msje_i, dynamic_slice_in_dim(dYdX_L2_norms, end, batch_size)))
	
@jit
def normalize_values(scores, normalizer): #store L2NormY, L2NormdYdX
	return scores/normalizer

def compute_loss_metrics(nn, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms, n_batches):
	#fill an array jax
	mse = 0.
	msje = 0.
	rel_mse = 0.
	rel_msje = 0.
	# errors = jnp.zeros((4,1))
	end = 0
	for _ in range(n_batches):
		end, a,b,c,d = mean_h1_seminorm_l2_errors_and_norms(nn, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms, end, batch_size)
		mse+=a
		msje += b
		rel_mse +=c
		rel_msje += d
		# mse += one_over_n_batches*jnp.mean(mse_i)
		# msje += one_over_n_batches*jnp.mean(msje_i)
		# rel_mse += one_over_n_batches*jnp.mean(normalize_values(mse_i, Y_batch_L2_norms))
		# rel_msje += one_over_n_batches*jnp.mean(normalize_values(msje_i, dYdX_batch_L2_norms))
		
	acc_l2 = 1. - jnp.sqrt(rel_mse)
	acc_h1 = 1. - jnp.sqrt(rel_msje)
	mean_h1_seminorm_loss = mse + msje
	return 1. - jnp.sqrt(rel_mse), 1. - jnp.sqrt(rel_msje), mse + msje

# one_over_n_batches = 1./n_batches
from jax import lax



@jit #move 
def compute_metrics(state, params, data, batch_size = 32):

	n_data = data['m'].shape[0]

	n_batches, remainder = divmod(n_data, batch_size)
	l2_errs = None 
	l2_normalizations = None

	h1_errs = None
	h1_normalizations = None

	for i_batch in range(n_batches):
		start = i_batch * batch_size 
		end = start + batch_size
		batch = jax.tree_map(lambda x : x[start:end], data) # no shuffling needed here
		l2_errs_i, l2_normalizations_i, h1_errs_i, h1_normalizations_i = compute_batched_errs(state,batch,params)
		if l2_errs is None:
			l2_errs = jnp.copy(l2_errs_i)
			l2_normalizations = jnp.copy(l2_normalizations_i)
			h1_errs = jnp.copy(h1_errs_i)
			h1_normalizations = jnp.copy(h1_normalizations_i)
		else:
			l2_errs = jnp.concatenate([l2_errs,l2_errs_i])
			l2_normalizations = jnp.concatenate([l2_normalizations,l2_normalizations_i])
			h1_errs = jnp.concatenate([h1_errs,h1_errs_i])
			h1_normalizations = jnp.concatenate([h1_normalizations,h1_normalizations_i])
	# Remainder
	if remainder > 0:
		batch = jax.tree_map(lambda x : x[end:end+remainder], data)
		l2_errs_i, l2_normalizations_i, h1_errs_i, h1_normalizations_i = compute_batched_errs(state,batch,params)
		l2_errs = jnp.concatenate([l2_errs,l2_errs_i])
		l2_normalizations = jnp.concatenate([l2_normalizations,l2_normalizations_i])
		h1_errs = jnp.concatenate([h1_errs,h1_errs_i])
		h1_normalizations = jnp.concatenate([h1_normalizations,h1_normalizations_i])

	l2_rel_squared_errors = jnp.divide(l2_errs,l2_normalizations)
	h1_rel_squared_errors = jnp.divide(h1_errs,h1_normalizations)

	l2_rms_rel_error = jnp.sqrt(jnp.mean(l2_rel_squared_errors,axis = 0))
	acc_l2 = 1. - l2_rms_rel_error

	h1_rms_rel_error = jnp.sqrt(jnp.mean(h1_rel_squared_errors,axis = 0))
	acc_h1 = 1. - h1_rms_rel_error

	loss = jnp.mean(l2_errs,axis=0) + jnp.mean(h1_errs,axis=0)

	return {'acc_l2': acc_l2, 'acc_h1':acc_h1, 'loss': loss}


def create_permuted_arrays(arrays, batch_size, *, key):
	dataset_size = arrays[0].shape[0]
	perm = jr.permutation(key, jnp.arange(dataset_size))
	(key,) = jr.split(key, 1)
	permuted_arrays = [array[perm] for array in arrays]
	return key, permuted_arrays

################################################################################
# Setup for training
metrics_history = {'train_loss': [],
					 'train_accuracy_l2': [],
					 'train_accuracy_h1': [],
					 'test_loss': [],
					 'test_accuracy_l2': [],
					 'test_accuracy_h1': [],
					 'epoch_time': [],
					 'epoch': []}

# def loss_fn(nn, X_batch, Y_batch, dYdX_batch):
# 	q_pred = vmap(nn)(X_batch)
# 	J_pred = vmap(jax.jacfwd(nn))(X_batch)
# 	loss = dj.l2_loss(q_pred, Y_batch) + dj.f_loss(J_pred, dYdX_batch)
# 	return loss

	# jnp.mean(np.sum(optax.l2_loss(predicted_Y, actual_Y),axis=1),axis=0) + jnp.mean(np.sum(optax.l2_loss(predicted_dYdX, actual_dYdX),axis=(1,2)),axis=0)


	
# train_batch_its = jnp.arange(n_train_batches)
# test_batch_its = jnp.arange(n_test_batches)

train_data_tuple = (m_train, u_train, J_train) #, u_norms_train, J_norms_train)
test_data_tuple = (m_test, u_test, J_test) #, u_norms_test, J_norms_test)




# Run the training loop


@eqx.filter_jit
def slice_data(input_X, actual_Y, actual_dYdX, batch_size, end):
	return (end + batch_size, 
		dynamic_slice_in_dim(input_X, end, batch_size),
		dynamic_slice_in_dim(actual_Y, end, batch_size), 
		dynamic_slice_in_dim(actual_dYdX, end, batch_size))

for epoch in range(1,args.num_epochs+1):
	loader_key, train_data_tuple = create_permuted_arrays(train_data_tuple, batch_size, key=loader_key) #, Y_norms_data, dYdX_norms_data
	
	X_data, Y_data, dYdX_data, _, _ = train_data_tuple
	# start_time = time.time()									
	end = 0
	for _ in range(n_train_batches):
		end, X_data_batch, Y_data_batch, dYdX_data_batch = slice_data(
			X_data, Y_data, dYdX_data,  batch_size, end)
		optimizer_state, nn  = take_step(optimizer_state,
										nn,
										X_data_batch,
										Y_data_batch,
										dYdX_data_batch) #loss,
		# can we combine Y and dYdX (flattened) into Y_dYdX_data
	epoch_time = time.time() - start_time

	
	# print(epoch) #, "loss", loss)#, loss_fn(nn, X_batch, Y_batch, dYdX_batch))

	# print('The last epoch took', epoch_time, 's')

	# metric_t0 = time.time()


	# Post-process and compute metrics after each epoch
	training_data_metrics = compute_loss_metrics(nn, *train_data_tuple, n_train_batches)

	loader_key, test_data_tuple   = create_permuted_arrays(test_data_tuple, batch_size, key=loader_key)
	
	compute_loss_metrics(nn, *test_data_tuple, n_test_batches)
	
	#testing_data_metrics

	# metrics_train = compute_metrics(train_state,params,training_data)
	# metrics_test = compute_metrics(train_state,params,testing_data)
	metric_time = time.time() - metric_t0

	metrics_history['train_loss'].append(np.array(training_data_metrics['loss']))
	metrics_history['train_accuracy_l2'].append(np.array(training_data_metrics['acc_l2']))
	metrics_history['train_accuracy_h1'].append(np.array(training_data_metrics['acc_h1']))
	metrics_history['test_loss'].append(np.array(test_data_metrics['loss']))
	metrics_history['test_accuracy_l2'].append(np.array(test_data_metrics['acc_l2']))
	metrics_history['test_accuracy_h1'].append(np.array(test_data_metrics['acc_h1']))
	metrics_history['epoch_time'].append(epoch_time)
	metrics_history['epoch'].append(epoch)


	print(f"train epoch: {epoch}, "
				f"loss: {metrics_history['train_loss'][-1]}, "
				f"accuracy l2 : {metrics_history['train_accuracy_l2'][-1] * 100}, "
				f"accuracy h1 : {metrics_history['train_accuracy_h1'][-1] * 100}")
	print(f"test epoch: {epoch}, "
				f"loss: {metrics_history['test_loss'][-1]}, "
				f"accuracy l2: {metrics_history['test_accuracy_l2'][-1] * 100}, "
				f"accuracy h1: {metrics_history['test_accuracy_h1'][-1] * 100}")
	print('Max test accuracy L2 = ',100*np.max(np.array(metrics_history['test_accuracy_l2'])))
	print('Max test accuracy H1 (semi-norm) = ',100*np.max(np.array(metrics_history['test_accuracy_h1'])))
	print('The metrics took', metric_time, 's')

