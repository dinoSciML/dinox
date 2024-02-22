import jax
import jax.numpy as jnp

def squared_l2_error(y_true, y_pred):
    return squared_l2_norm(y_true - y_pred)

def squared_l2_norm(y):
    return jnp.inner(y, y)

# def normalized_l2(y_true,y_pred):
# 	return jnp.mean(jax.vmap(squared_l2_norm)(y_true - y_pred),axis=0)/\
# 			(jnp.mean(jax.vmap(squared_l2_norm)(y_true),axis=0)+1e-4)

def l2_loss(y_true,y_pred):
	return jnp.mean(jax.vmap(squared_l2_norm)(y_true - y_pred),axis=0)

def mse(y_true_batched, y_pred_batched):
    return jnp.mean(jax.vmap(squared_l2_error)(y_true_batched, y_pred_batched), axis=0)


def squared_f_error(y_true, y_pred):
    return squared_f_norm(y_true - y_pred)

def squared_f_norm(y):
    return jnp.sum(jnp.square(y))


# def normalized_f(y_true,y_pred):
# 	return jnp.mean(jax.vmap(squared_f_norm)(y_true - y_pred), axis=0)/\
# 			(jnp.mean(jax.vmap(squared_f_norm)(y_true), axis=0) )

def f_loss(y_true,y_pred):
	return jnp.mean(jax.vmap(squared_f_norm)(y_true - y_pred), axis=0)

