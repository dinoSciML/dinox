import sys, os
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', '--batch_size', type = int, default = 32, help = 'gradient batch size')
parser.add_argument('-step_size', '--step_size', type=float, default=1e-3, help="What step size or 'learning rate'?")
parser.add_argument('-num_epochs', '--num_epochs', type=int, default=100, help="How many epochs")
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

import numpy as np
import time
import functools

import jax 
import jax.numpy as jnp 
import jax.random as random 
import optax 

from flax.training import train_state

import dinojax as dj

# Load the data
data_file = 'mq_data_reduced.npz'
data = np.load(args.data_dir+data_file)

m_data = data['m_data']
u_data = data['q_data']
J_data = data['J_data']


assert m_data.shape[0] == u_data.shape[0] == J_data.shape[0]

n_data, dM = m_data.shape
_, dU = u_data.shape

n_train = 800

# Fix me! This should be separate loaded files for test and train!!!!!
training_data = { 'm' : m_data[:n_train], 'u' : u_data[:n_train],'J':J_data[:n_train]}

testing_data = { 'm' : m_data[n_train:], 'u' : u_data[n_train:],'J':J_data[n_train:]}



batch_size = args.batch_size 
num_epochs = args.num_epochs

n_batches, remainder = divmod(n_train, batch_size)

train_steps_per_epoch = n_batches + int(bool(remainder))
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

network = dj.GenericDense(layer_widths=nn_widths, activation='gelu', output_size=dU)

network = dj.DINO(network)

rng_key = jax.random.PRNGKey(0)

@jax.jit
def initialize(params_rng):
	init_rngs = {'params': params_rng}
	input_shape = (1, dM)
	variables = network.init(init_rngs, jnp.ones(input_shape, jnp.float32))
	return variables

def train_step_dino():
    # @jax.jit
    def _train_step_dino(state, batch):
        """ Train for a single step """ 
        def loss_fn(params):
            q_pred, J_pred = state.apply_fn(params, batch['m'])
            loss = dj.l2_loss(q_pred, batch['u']) + dj.f_loss(J_pred,batch['J'])
            return loss
        
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state 
    return _train_step_dino

params = initialize(rng_key)

rng = jax.random.PRNGKey(args.run_seed)

train_state = train_state.TrainState.create(apply_fn=network.apply_fn, params=params, tx=optimizer)

train_step = train_step_dino()


def compute_batched_errs(state, batch, params):
	y_preds, J_preds = state.apply_fn(state.params, batch['m'])
	y_true = batch['u']
	J_true = batch['J']

	l2_errs = jax.vmap(dj.squared_l2_error)(y_true, y_preds)
	l2_normalizations = jax.vmap(dj.squared_l2_norm)(y_true)

	h1_errs = jax.vmap(dj.squared_f_error)(J_true, J_preds)
	h1_normalizations = jax.vmap(dj.squared_f_norm)(J_true)

	return l2_errs,l2_normalizations, h1_errs, h1_normalizations

@jax.jit
def compute_metrics(state, params, data, batch_size = 32):
	losses = []
	l2_accs = []
	h1_accs = []

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

for epoch in range(1,args.num_epochs):
	rng_key, subkey = jax.random.split(rng_key)
	shuffled_inds = jax.random.permutation(subkey, n_train, axis=0)
	start_time = time.time()

	for i_batch in range(n_batches):
		start = i_batch * batch_size 
		end = start + batch_size
		batch = jax.tree_map(lambda x : x[shuffled_inds[start:end]], training_data)
		# params = optimizer.update(params,batch)
		train_state = train_step(train_state, batch)

	# Remainder
	if remainder > 0:
		batch = jax.tree_map(lambda x : x[shuffled_inds[end:end+remainder]], training_data)
		# params = optimizer.update(params,batch)
		train_state = train_step(train_state, batch)

	epoch_time = time.time() - start_time
	print('The last epoch took', epoch_time, 's')

	metric_t0 = time.time()
	# Post-process and compute metrics after each epoch
	metrics_train = compute_metrics(train_state,params,training_data)
	metrics_test = compute_metrics(train_state,params,testing_data)
	metric_time = time.time() - metric_t0

	metrics_history['train_loss'].append(np.array(metrics_train['loss']))
	metrics_history['train_accuracy_l2'].append(np.array(metrics_train['acc_l2']))
	metrics_history['train_accuracy_h1'].append(np.array(metrics_train['acc_h1']))
	metrics_history['test_loss'].append(np.array(metrics_test['loss']))
	metrics_history['test_accuracy_l2'].append(np.array(metrics_test['acc_l2']))
	metrics_history['test_accuracy_h1'].append(np.array(metrics_test['acc_h1']))
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

