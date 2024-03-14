import equinox as eqx
import jax
from jax.lax import dynamic_slice_in_dim
import jax.numpy as jnp
from jax import vjp, vmap
import optax
from typing import Callable

@jax.jit
def squared_l2_error(y_true, y_pred):
    return squared_l2_norm(y_true - y_pred)

@jax.jit
def squared_l2_norm(y):
    return jnp.inner(y, y)

# def normalized_l2(y_true,y_pred):
# 	return jnp.mean(jax.vmap(squared_l2_norm)(y_true - y_pred),axis=0)/\
# 			(jnp.mean(jax.vmap(squared_l2_norm)(y_true),axis=0)+1e-4)

def l2_loss(y_true,y_pred):
	return jnp.mean(jax.vmap(squared_l2_norm)(y_true - y_pred),axis=0)

def mse(y_true_batched, y_pred_batched):
    return jnp.mean(jax.vmap(squared_l2_error)(y_true_batched, y_pred_batched), axis=0)

@jax.jit
def squared_f_error(y_true, y_pred):
    return squared_f_norm(y_true - y_pred)

@jax.jit
def squared_f_norm(y):
    return jnp.sum(jnp.square(y))


# def normalized_f(y_true,y_pred):
# 	return jnp.mean(jax.vmap(squared_f_norm)(y_true - y_pred), axis=0)/\
# 			(jnp.mean(jax.vmap(squared_f_norm)(y_true), axis=0) )

def f_loss(y_true,y_pred):
	return jnp.mean(jax.vmap(squared_f_norm)(y_true - y_pred), axis=0)

def value_and_jacrev(f, xs):
    _, pullback =  vjp(f, xs[0])
    basis = jnp.eye(_.size, dtype=_.dtype)

    @jax.jit
    def value_and_jacrev_x(x):
        y, pullback = vjp(f, x)
        jac = vmap(pullback)(basis)
        return y, jac[0] # There is only one jacobian matrix here, so we extract it
    return vmap(value_and_jacrev_x)(xs)

def create_mean_h1_seminorm_l2_errors_and_norms(dM):
	@eqx.filter_jit
	def mean_h1_seminorm_l2_errors_and_norms(nn, X, Y, dYdX):
		predicted_Y, predicted_dYdX =  value_and_jacrev(nn, dynamic_slice_in_dim(X, end_idx, batch_size))
		# predicted_Y = predicted_Y.squeeze()
		# predicted_dYdX = predicted_dYdX.squeeze()
		# batch_se  = jnp.mean(optax.l2_loss(predicted_Y.squeeze(), Y),axis=1)
		# batch_sje = jnp.mean(optax.l2_loss(predicted_dYdX.squeeze(), dYdX)*dM,axis=(1,2))
		mse_i, msje_i = jnp.mean(optax.l2_loss(predicted_Y.squeeze(), dynamic_slice_in_dim(Y, end_idx, batch_size)),axis=1), jnp.mean(optax.l2_loss(predicted_dYdX.squeeze(),
		dynamic_slice_in_dim(dYdX, end_idx, batch_size))*dM,axis=(1,2))
		return  end_idx + batch_size, one_over_n_batches*jnp.mean(mse_i),
		one_over_n_batches*jnp.mean(mse_i),
		one_over_n_batches*jnp.mean(mse_i), one_over_n_batches*jnp.mean(normalize_values(mse_i, dynamic_slice_in_dim(Y_L2_norms, end_idx, batch_size))), one_over_n_batches*jnp.mean(normalize_values(msje_i, dynamic_slice_in_dim(dYdX_L2_norms, end_idx, batch_size)))
	return mean_h1_seminorm_l2_errors_and_norms

@eqx.filter_jit
def mean_l2_norm_errors_and_norms(nn, X, Y, dYdX):
	predicted_Y, predicted_dYdX =  value_and_jacrev(nn, dynamic_slice_in_dim(X, end_idx, batch_size))
	# predicted_Y = predicted_Y.squeeze()
	# predicted_dYdX = predicted_dYdX.squeeze()
	# batch_se  = jnp.mean(optax.l2_loss(predicted_Y.squeeze(), Y),axis=1)
	# batch_sje = jnp.mean(optax.l2_loss(predicted_dYdX.squeeze(), dYdX)*dM,axis=(1,2))
	mse_i, msje_i = jnp.mean(optax.l2_loss(predicted_Y.squeeze(), dynamic_slice_in_dim(Y, end_idx, batch_size)),axis=1), jnp.mean(optax.l2_loss(predicted_dYdX.squeeze(),
	dynamic_slice_in_dim(dYdX, end_idx, batch_size))*dM,axis=(1,2))
	return  end_idx + batch_size, one_over_n_batches*jnp.mean(mse_i),
	one_over_n_batches*jnp.mean(mse_i),
	one_over_n_batches*jnp.mean(mse_i), one_over_n_batches*jnp.mean(normalize_values(mse_i, dynamic_slice_in_dim(Y_L2_norms, end_idx, batch_size))), one_over_n_batches*jnp.mean(normalize_values(msje_i, dynamic_slice_in_dim(dYdX_L2_norms, end_idx, batch_size)))

def create_mean_h1_seminorm_loss(dM: int) -> Callable:
	@eqx.filter_jit
	def mean_h1_seminorm_loss(nn:eqx.nn, input_X: jax.Array, actual_Y: jax.Array, actual_dYdX: jax.Array):
		predicted_Y, predicted_dYdX =  value_and_jacrev(nn, input_X)
		return jnp.mean(optax.l2_loss(predicted_Y.squeeze(), actual_Y)) + jnp.mean(optax.l2_loss(predicted_dYdX.squeeze(), actual_dYdX))*dM
	return mean_h1_seminorm_loss

@eqx.filter_jit
def mean_l2_norm_loss(nn:eqx.nn, input_X: jax.Array, actual_Y: jax.Array):
	predicted_Y = nn(input_X)
	return jnp.mean(optax.l2_loss(predicted_Y.squeeze(), actual_Y))

create_grad_mean_h1_seminorm_loss = lambda dM: eqx.filter_grad(create_mean_h1_seminorm_loss(dM))
grad_mean_l2_norm_loss = eqx.filter_grad(mean_l2_norm_loss)

@jax.jit
def normalize_values(scores, normalizers): #store L2NormY, L2NormdYdX
	return scores/normalizers

def compute_l2_loss_metrics(nn, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms, n_batches):
#fill an array jax
	mse = 0.
	msje = 0.
	rel_mse = 0.
	rel_msje = 0.
	# errors = jnp.zeros((4,1))
	end_idx = 0
	for _ in range(n_batches):
		end_idx, a,b,c,d = mean_l2_norm_errors_and_norms(nn, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms, end_idx, batch_size)
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
def create_compute_h1_loss_metrics(dM: int) -> Callable:
	mean_h1_seminorm_errors_and_norms = create_mean_h1_seminorm_l2_errors_and_norms(dM)
	def compute_loss_metrics(nn, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms, n_batches):
		#DOCUMENT ME

		#fill an array jax
		mse = 0.
		msje = 0.
		rel_mse = 0.
		rel_msje = 0.
		# errors = jnp.zeros((4,1))
		end_idx = 0
		for _ in range(n_batches):
			end_idx, a,b,c,d = mean_h1_seminorm_errors_and_norms(nn, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms, end_idx, batch_size)
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

# def loss_fn(nn, X_batch, Y_batch, dYdX_batch):
# 	q_pred = vmap(nn)(X_batch)
# 	J_pred = vmap(jax.jacfwd(nn))(X_batch)
# 	loss = dj.l2_loss(q_pred, Y_batch) + dj.f_loss(J_pred, dYdX_batch)
# 	return loss

	# jnp.mean(np.sum(optax.l2_loss(predicted_Y, Y),axis=1),axis=0) + jnp.mean(np.sum(optax.l2_loss(predicted_dYdX, dYdX),axis=(1,2)),axis=0)


# def compute_metrics_old(state, params, data, batch_size = 32):
# 	n_data = data['m'].shape[0]
# 	for i_batch in range(n_batches):
# 		start = i_batch * batch_size 
# 		end_idx = start + batch_size
# 		batch = jax.tree_map(lambda x : x[start:end_idx], data) # no shuffling needed here
# 		l2_errs_i, l2_normalizations_i, h1_errs_i, h1_normalizations_i = compute_batched_errs(state,batch,params)
# 		if l2_errs is None:
# 			l2_errs = jnp.copy(l2_errs_i)
# 			l2_normalizations = jnp.copy(l2_normalizations_i)
# 			h1_errs = jnp.copy(h1_errs_i)
# 			h1_normalizations = jnp.copy(h1_normalizations_i)
# 		else:
# 			l2_errs = jnp.concatenate([l2_errs,l2_errs_i])
# 			l2_normalizations = jnp.concatenate([l2_normalizations,l2_normalizations_i])
# 			h1_errs = jnp.concatenate([h1_errs,h1_errs_i])
# 			h1_normalizations = jnp.concatenate([h1_normalizations,h1_normalizations_i])

# 	l2_rel_squared_errors = jnp.divide(l2_errs,l2_normalizations)
# 	h1_rel_squared_errors = jnp.divide(h1_errs,h1_normalizations)

# 	l2_rms_rel_error = jnp.sqrt(jnp.mean(l2_rel_squared_errors,axis = 0))
# 	acc_l2 = 1. - l2_rms_rel_error

# 	h1_rms_rel_error = jnp.sqrt(jnp.mean(h1_rel_squared_errors,axis = 0))
# 	acc_h1 = 1. - h1_rms_rel_error

# 	loss = jnp.mean(l2_errs,axis=0) + jnp.mean(h1_errs,axis=0)

# 	return {'acc_l2': acc_l2, 'acc_h1':acc_h1, 'loss': loss}