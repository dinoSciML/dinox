from typing import Callable

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from equinox import filter_grad, filter_jit, filter_vmap
from jax import jit, vjp

from .equinox_nn_factories import eqxModule

__squared_norm = lambda x: jnp.sum(x**2)
__vectorized_squared_norm = jax.vmap(jit(__squared_norm, device=jax.devices("gpu")[0]), in_axes=0)
__cpu_vectorized_squared_norm = jax.vmap(jit(__squared_norm, device=jax.devices("cpu")[0]), in_axes=0)
__divide = jit(lambda scores, normalizations: scores / normalizations, device=jax.devices("gpu")[0])
__cpu_divide = jit(
    lambda scores, normalizations: scores / normalizations,
    device=jax.devices("cpu")[0],
)
# This is only an Monte Carlo estimate of the L2 Bochner loss, i.e. just an empirical MSE.
L2_Bochner_loss: Callable[[eqxModule, jax.Array, jax.Array], float] = filter_jit(
    lambda nn, input_X, actual_Y: jnp.mean(jnp.sum((filter_vmap(nn)(input_X) - actual_Y) ** 2, axis=1))
)
vectorized_grad_L2_Bochner_loss = filter_grad(L2_Bochner_loss)


@filter_jit(backend="gpu")
def vectorized_H1_Bochner_loss(
    nn: eqxModule,
    input_X: jax.Array,
    actual_Y: jax.Array,
    actual_dYdX: jax.Array,
) -> float:
    predicted_Y, predicted_dYdX = __vectorized_value_and_jacrev(nn, input_X)
    return jnp.mean(
        __vectorized_squared_norm(actual_Y - predicted_Y) + __vectorized_squared_norm(actual_dYdX - predicted_dYdX)
    )


vectorized_grad_H1_Bochner_loss = filter_grad(vectorized_H1_Bochner_loss)


@filter_jit(backend="gpu")
def __vectorized_value_and_jacrev(f: Callable[[jax.Array], jax.Array], xs: jax.Array) -> jax.Array:
    """
    Compute function values and full Jacobians for a batch of inputs using vjp.

    Parameters:
        f (Callable[[jax.Array], jax.Array]): Function to evaluate.
        xs (jax.Array): Batch of inputs.

    Returns:
        jax.Array: Array of tuples (f(x), Jacobian at x).

    Notes:
        Uses JAX's vjp with an identity matrix and vmap for efficient batch processing.
    """
    _, pullback = vjp(f, xs[0])
    basis = jnp.eye(_.size, dtype=_.dtype)  # Create an identity matrix the size of the output

    def value_and_jacrev_x(x: jax.Array):
        """
        Helper function to compute function value and Jacobian at a given input.
        """
        y, pullback = vjp(f, x)
        return (
            y,
            filter_vmap(pullback)(basis)[0],
        )  # Extract the single Jacobian matrix from the vmap result

    return filter_vmap(value_and_jacrev_x)(xs)


@filter_jit(backend="cpu")
def __cpu_vectorized_value_and_jacrev(f: Callable[[jax.Array], jax.Array], xs: jax.Array) -> jax.Array:
    """
    Compute function values and full Jacobians for a batch of inputs using vjp.

    Parameters:
        f (Callable[[jax.Array], jax.Array]): Function to evaluate.
        xs (jax.Array): Batch of inputs.

    Returns:
        jax.Array: Array of tuples (f(x), Jacobian at x).

    Notes:
        Uses JAX's vjp with an identity matrix and vmap for efficient batch processing.
    """
    _, pullback = vjp(f, xs[0])
    basis = jnp.eye(_.size, dtype=_.dtype)  # Create an identity matrix the size of the output

    def value_and_jacrev_x(x: jax.Array):
        """
        Helper function to compute function value and Jacobian at a given input.
        """
        y, pullback = vjp(f, x)
        return (
            y,
            filter_vmap(pullback)(basis)[0],
        )  # Extract the single Jacobian matrix from the vmap result

    return filter_vmap(value_and_jacrev_x)(xs)


@filter_jit(backend="cpu")
def cpu_compute_H1_bochner_relative_errors(
    nn: eqxModule,
    X: jax.Array,
    fX: jax.Array,
    dfXdX: jax.Array,
    fX_L2_norms: jax.Array,
    dfXdX_Frobenius_norms: jax.Array,
) -> jax.Array:
    """
    Compute RMSE for outputs and Jacobians, plus their normalized errors.

    Parameters:
        nn: eqxModule - Model predicting outputs and Jacobians.
        X: jax.Array - Inputs.
        fX: jax.Array - Target outputs.
        dfXdX: jax.Array - Target Jacobians.
        fX_L2_norms: jax.Array - L2 norms of outputs.
        dfXdX_Frobenius_norms: jax.Array - Frobenius norms of Jacobians.

    Returns:
        jax.Array: Tuple (RMSE(NN(X);fX), RMSE(JacNN(X); dfXdX), normalized RMSE(NN), normalized RMSE(JacNN).
    """

    predicted_fX, predicted_dfXdX = __cpu_vectorized_value_and_jacrev(nn, X)
    Yhat_squared_norm_errors = __cpu_vectorized_squared_norm(predicted_fX - fX)
    dYhatdX_squared_norm_errors = __cpu_vectorized_squared_norm(predicted_dfXdX - dfXdX)

    return (
        jnp.sqrt(jnp.mean(Yhat_squared_norm_errors)),  #  Root mean (function) squared error
        jnp.sqrt(jnp.mean(dYhatdX_squared_norm_errors)),  # Root mean (Jacobian) squared error
        # Root mean relative (function) squared error
        jnp.sqrt(jnp.mean(__cpu_divide(Yhat_squared_norm_errors, fX_L2_norms))),
        # Root mean relative (Jacobian) squared error
        jnp.sqrt(jnp.mean(__cpu_divide(dYhatdX_squared_norm_errors, dfXdX_Frobenius_norms))),
    )


@filter_jit(backend="cpu")
def cpu_compute_L2_bochner_relative_errors(
    nn: eqxModule,
    X: jax.Array,
    fX: jax.Array,
    fX_L2_norms: jax.Array,
) -> jax.Array:
    """
    Compute RMSE for outputs and Jacobians, plus their normalized errors.

    Parameters:
        nn: eqxModule - Model predicting outputs and Jacobians.
        X: jax.Array - Inputs.
        fX: jax.Array - Target outputs.
        dfXdX: jax.Array - Target Jacobians.
        fX_L2_norms: jax.Array - L2 norms of outputs.
        dfXdX_Frobenius_norms: jax.Array - Frobenius norms of Jacobians.

    Returns:
        jax.Array: Tuple (RMSE(NN(X);fX), RMSE(JacNN(X); dfXdX), normalized RMSE(NN) normalized RMSE(JacNN).
    """
    Yhat_squared_norm_errors = __cpu_vectorized_squared_norm(filter_vmap(nn)(X) - fX)
    return (
        jnp.sqrt(jnp.mean(Yhat_squared_norm_errors)),  #  Root mean (function) squared error
        # Root mean relative (function) squared error
        jnp.sqrt(jnp.mean(__cpu_divide(Yhat_squared_norm_errors, fX_L2_norms))),
        # Root mean relative Jacobian squared error
    )


@filter_jit(backend="gpu")
def compute_H1_bochner_relative_errors(
    nn: eqxModule,
    X: jax.Array,
    fX: jax.Array,
    dfXdX: jax.Array,
    fX_L2_norms: jax.Array,
    dfXdX_Frobenius_norms: jax.Array,
) -> jax.Array:
    """
    Compute RMSE for outputs and Jacobians, plus their normalized errors.

    Parameters:
        nn: eqxModule - Model predicting outputs and Jacobians.
        X: jax.Array - Inputs.
        fX: jax.Array - Target outputs.
        dfXdX: jax.Array - Target Jacobians.
        fX_L2_norms: jax.Array - L2 norms of outputs.
        dfXdX_Frobenius_norms: jax.Array - Frobenius norms of Jacobians.

    Returns:
        jax.Array: Tuple (RMSE(NN(X);fX), RMSE(JacNN(X); dfXdX), normalized RMSE(NN) normalized RMSE(JacNN).
    """

    predicted_fX, predicted_dfXdX = __vectorized_value_and_jacrev(nn, X)
    Yhat_squared_norm_errors = __vectorized_squared_norm(predicted_fX - fX)
    dYhatdX_squared_norm_errors = __vectorized_squared_norm(predicted_dfXdX - dfXdX)

    return (
        jnp.sqrt(jnp.mean(Yhat_squared_norm_errors)),  #  Root mean (function) squared error
        jnp.sqrt(jnp.mean(dYhatdX_squared_norm_errors)),  # Root mean (Jacobian) squared error
        # Root mean relative (function) squared error
        jnp.sqrt(jnp.mean(__divide(Yhat_squared_norm_errors, fX_L2_norms))),
        # Root mean relative Jacobian squared error
        jnp.sqrt(jnp.mean(__divide(dYhatdX_squared_norm_errors, dfXdX_Frobenius_norms))),
    )


@filter_jit(backend="gpu")
def compute_L2_bochner_relative_errors(
    nn: eqxModule,
    X: jax.Array,
    fX: jax.Array,
    fX_L2_norms: jax.Array,
) -> jax.Array:
    """
    Compute RMSE for outputs and Jacobians, plus their normalized errors.

    Parameters:
        nn: eqxModule - Model predicting outputs and Jacobians.
        X: jax.Array - Inputs.
        fX: jax.Array - Target outputs.
        dfXdX: jax.Array - Target Jacobians.
        fX_L2_norms: jax.Array - L2 norms of outputs.
        dfXdX_Frobenius_norms: jax.Array - Frobenius norms of Jacobians.

    Returns:
        jax.Array: Tuple (RMSE(NN(X);fX), RMSE(JacNN(X); dfXdX), normalized RMSE(NN) normalized RMSE(JacNN).
    """
    Yhat_squared_norm_errors = __vectorized_squared_norm(filter_vmap(nn)(X) - fX)
    return (
        jnp.sqrt(jnp.mean(Yhat_squared_norm_errors)),  #  Root mean (function) squared error
        # Root mean relative (function) squared error
        jnp.sqrt(jnp.mean(__divide(Yhat_squared_norm_errors, fX_L2_norms))),
        # Root mean relative Jacobian squared error
    )
