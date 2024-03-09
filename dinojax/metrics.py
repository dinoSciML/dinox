import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vjp, vmap

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

    @jit
    def value_and_jacrev_x(x):
        y, pullback = vjp(f, x)
        jac = vmap(pullback)(basis)
        return y, jac[0] #
    return vmap(value_and_jacrev_x)(xs)

@eqx.filter_jit
def mean_h1_seminorm_loss(nn:eqx.nn, input_X: jax.Array, actual_Y: jax.Array, actual_dYdX: jax.Array):
	predicted_Y, predicted_dYdX =  value_and_jacrev(nn, input_X)
	# predicted_Y = predicted_Y.squeeze()
	# predicted_dYdX = predicted_dYdX.squeeze()
	return jnp.mean(optax.l2_loss(predicted_Y.squeeze(), actual_Y)) + jnp.mean(optax.l2_loss(predicted_dYdX.squeeze(), actual_dYdX))*dM

grad_mean_h1_norm_loss_fn = eqx.filter_grad(mean_h1_norm_loss)
