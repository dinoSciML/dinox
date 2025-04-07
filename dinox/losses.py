from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vjp, vmap

# losses.py
squared_norm = lambda x: jnp.sum(x**2)
vectorized_squared_norm = jax.vmap(jax.jit(squared_norm), in_axes=0)
cpu_vectorized_squared_norm = jax.vmap(
    jax.jit(squared_norm, device=jax.devices("cpu")[0]), in_axes=0
)
divide = jax.jit(lambda scores, normalizations: scores / normalizations)
cpu_divide = jax.jit(
    lambda scores, normalizations: scores / normalizations, device=jax.devices("cpu")[0]
)
# This is only an Monte Carlo estimate of the L2 Bochner loss, i.e. just an empirical MSE.
L2_Bochner_loss: Callable[[eqx.Module, jax.Array, jax.Array], float] = eqx.filter_jit(
    lambda nn, input_X, actual_Y: jnp.mean(
        jnp.sum((vmap(nn)(input_X) - actual_Y) ** 2, axis=1)
    )
)  # cpu version?
vectorized_grad_L2_Bochner_loss = eqx.filter_grad(L2_Bochner_loss)  # cpu_ version?


@eqx.filter_jit
def vectorized_H1_Bochner_loss(
    nn: eqx.Module, input_X: jax.Array, actual_Y: jax.Array, actual_dYdX: jax.Array
) -> float:
    predicted_Y, predicted_dYdX = vectorized_value_and_jacrev(nn, input_X)
    return jnp.mean(
        vectorized_squared_norm(actual_Y - predicted_Y)
        + vectorized_squared_norm(actual_dYdX - predicted_dYdX)
    )


vectorized_grad_H1_Bochner_loss = eqx.filter_grad(vectorized_H1_Bochner_loss)


def vectorized_value_and_jacrev(
    f: Callable[[jax.Array], jax.Array], xs: jax.Array
) -> jax.Array:
    """
    Computes the function value and Jacobian for each input in a batch using vector-Jacobian product (vjp).

    Parameters
    ----------
    f : Callable[[jax.Array], jax.Array]
        The function for which the values and Jacobians are computed. This function should accept
        a JAX array as input and return a JAX array as output.
    xs : jax.Array
        An array of inputs to the function `f`. The function `f` will be evaluated at each of these inputs.

    Returns
    -------
    jax.Array
        An array of tuples, where each tuple contains the function value at the corresponding input
        and the Jacobian matrix of the function at that input. The Jacobian matrix is provided for each
        input independently.

    Notes
    -----
    This function uses JAX's automatic differentiation capabilities via the `vjp` function, which returns
    a function (`pullback`) that computes the vector-Jacobian product for a given vector. Here, the identity
    matrix is used with `vmap` to effectively compute the full Jacobian matrix for each input.

    The `vmap` function is used to vectorize `value_and_jacrev_x` across all inputs in `xs`, thereby
    enabling efficient batch processing of the function and its Jacobian computation. The function assumes
    that all inputs in `xs` can be processed independently and in parallel, which is typical for many
    scientific computing and machine learning applications.

    Each element in `results` will be a tuple, with the first element being the squared value of the input,
    and the second element being the Jacobian at that point, all encapsulated in a Jacobian matrix.
    """
    _, pullback = vjp(f, xs[0])
    basis = jnp.eye(
        _.size, dtype=_.dtype
    )  # Create an identity matrix the size of the output

    def value_and_jacrev_x(x: jax.Array):
        """
        Helper function to compute function value and Jacobian at a given input.

        Parameters
        ----------
        x : jax.Array
            Single input to the function `f`.

        Returns
        -------
        Tuple[jax.Array, jax.Array]
            A tuple containing the function value and its Jacobian matrix at the input `x`.
        """
        y, pullback = vjp(f, x)
        return (
            y,
            vmap(pullback)(basis)[0],
        )  # Extract the single Jacobian matrix from the vmap result

    return vmap(value_and_jacrev_x)(xs)


@eqx.filter_jit(backend="cpu")
def cpu_compute_bochner_relative_errors(
    nn: eqx.Module,
    X: jax.Array,
    Y: jax.Array,
    dYdX: jax.Array,
    Y_L2_norms: jax.Array,
    dYdX_Frobenius_norms: jax.Array,
) -> jax.Array:
    """
    Computes mean squared errors for a flat structure of predicted and actual values
    and their Jacobians over a batch, and normalizes these errors by the L2 norms of
    the actual values and Jacobians.

    Parameters
    ----------
    nn : eqx.Module
        The neural network model that predicts both values and their Jacobians.
    X : jax.Array
        The input values
    Y : jax.Array
        The array of output values for comparison.
    dYdX: jax. Array
        The array of Jacobians
    Y_L2_norms : jax.Array
        The L2 norms of the output values.
    dYdX_Frobenius_norms : jax.Array
        The Frobenius norms of the Jacobians.

    Returns
    -------
    jax.Array
        An array containing the mean squared errors for the outputs and Jacobians,
        as well as their respective normalized mean squared errors, calculated as:
        [mean MSE for Y, mean MSE for dYdX, normalized MSE for Y, normalized MSE for dYdX].

    Notes
    -----
    This function leverages JAX's `vmap` to vectorize the computation of squared differences,
    applies a lambda function to compute squared errors, and splits the result to handle
    outputs and Jacobians separately. This ensures efficient batch processing and is
    particularly useful for gradient-based optimization where sensitivity information is crucial.
    """

    predicted_Y, predicted_dYdX = vectorized_value_and_jacrev(
        nn, X
    )  # cpu?? ensure nn and X are on cpu
    Yhat_squared_norm_errors = cpu_vectorized_squared_norm(predicted_Y - Y)
    dYhatdX_squared_norm_errors = cpu_vectorized_squared_norm(predicted_dYdX - dYdX)

    return (
        jnp.sqrt(
            jnp.mean(Yhat_squared_norm_errors)
        ),  #  Root mean (function) squared error
        jnp.sqrt(
            jnp.mean(dYhatdX_squared_norm_errors)
        ),  # Root mean (Jacobian) squared error
        jnp.sqrt(
            jnp.mean(cpu_divide(Yhat_squared_norm_errors, Y_L2_norms))
        ),  # Root mean relative (function) squared error
        jnp.sqrt(
            jnp.mean(cpu_divide(dYhatdX_squared_norm_errors, dYdX_Frobenius_norms))
        ),  # Root mean relative Jacobian squared error
    )
