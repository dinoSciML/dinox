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

from typing import Any, Callable, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import vjp, vmap
from jax.lax import dynamic_slice_in_dim as jittable_slice

__all__ = [
    "grad_mean_l2_norm_loss",
    "grad_mean_h1_norm_loss_flattened",
    "compute_batched_flattened_h1_loss_metrics",
    "compute_flattened_h1_loss_metrics",
    "compute_l2_loss_metrics",
    "mean_h1_seminorm_and_l2_errors_and_rel_errors_flat",
]


# # @eqx.filter_jit
# def __value_and_jacrev(f: Callable[[jax.Array], jax.Array], xs: jax.Array) -> jax.Array:
#     """
#     Computes the function value and Jacobian for each input in a batch using vector-Jacobian product (vjp).

#     Parameters
#     ----------
#     f : Callable[[jax.Array], jax.Array]
#         The function for which the values and Jacobians are computed. This function should accept
#         a JAX array as input and return a JAX array as output.
#     xs : jax.Array
#         An array of inputs to the function `f`. The function `f` will be evaluated at each of these inputs.

#     Returns
#     -------
#     jax.Array
#         An array of tuples, where each tuple contains the function value at the corresponding input
#         and the Jacobian matrix of the function at that input. The Jacobian matrix is provided for each
#         input independently.

#     Notes
#     -----
#     This function uses JAX's automatic differentiation capabilities via the `vjp` function, which returns
#     a function (`pullback`) that computes the vector-Jacobian product for a given vector. Here, the identity
#     matrix is used with `vmap` to effectively compute the full Jacobian matrix for each input.

#     The `vmap` function is used to vectorize `value_and_jacrev_x` across all inputs in `xs`, thereby
#     enabling efficient batch processing of the function and its Jacobian computation. The function assumes
#     that all inputs in `xs` can be processed independently and in parallel, which is typical for many
#     scientific computing and machine learning applications.

#     Example
#     -------
#     Consider a simple quadratic function `f(x) = x ** 2`. To compute the function value and its Jacobian
#     for a range of values:

#         f = lambda x: x ** 2
#         xs = jnp.array([1.0, 2.0, 3.0])
#         results = __value_and_jacrev(f, xs)

#     Each element in `results` will be a tuple, with the first element being the squared value of the input,
#     and the second element being the Jacobian at that point, all encapsulated in a Jacobian matrix.
#     """
#     _, pullback = vjp(f, xs[0])
#     basis = jnp.eye(
#         _.size, dtype=_.dtype
#     )  # Create an identity matrix the size of the output

#     def value_and_jacrev_x(x: jax.Array):
#         """
#         Helper function to compute function value and Jacobian at a given input.

#         Parameters
#         ----------
#         x : jax.Array
#             Single input to the function `f`.

#         Returns
#         -------
#         Tuple[jax.Array, jax.Array]
#             A tuple containing the function value and its Jacobian matrix at the input `x`.
#         """
#         y, pullback = vjp(f, x)
#         jac = vmap(pullback)(basis)
#         return y, jac[0]  # Extract the single Jacobian matrix from the vmap result

#     return vmap(value_and_jacrev_x)(xs)


@eqx.filter_jit
def __value_and_jacrev_flattened(f: Callable, xs: jax.Array) -> jax.Array:
    """
    Computes the value and flattened Jacobian (Jacobian vector product) of a function
    for each item in an input array.

    Parameters
    ----------
    f : Callable
        The function for which the value and Jacobian are to be computed. This function
        should take a single JAX array as input and return a JAX array as output.
    xs : jax.Array
        An array of inputs on which the function `f` is evaluated. Each item in `xs`
        should be suitable as input to the function `f`.

    Returns
    -------
    jax.Array
        An array where each element is the concatenation of the function value at the corresponding
        input and the flattened Jacobian of the function at that input. The concatenation is done such
        that each function output is followed by its entire flattened Jacobian vector.

    Notes
    -----
    This function utilizes JAX's automatic differentiation capabilities. The `vjp` (vector-Jacobian product)
    is used to compute the Jacobian with respect to the input. This approach is particularly useful when
    dealing with functions where both the output value and the gradient (or sensitivity information) at each
    input are required simultaneously.

    The computation is vectorized over the input array `xs` using `vmap`, which efficiently handles
    batch evaluations of the `value_and_jacrev_x_flattened` sub-function without explicit Python loops,
    thereby leveraging JAX's just-in-time compilation provided by `eqx.filter_jit` to optimize the operation.

    Example
    -------
    Given a function `f` that computes `x**2`, and an input array `xs`:

        f = lambda x: x ** 2
        xs = jnp.array([1.0, 2.0, 3.0])

    The output will be an array where each element is `f(x)` followed by `df/dx` evaluated at each `x`.
    """
    _, pullback = vjp(f, xs[0])
    basis = jnp.eye(
        _.size, dtype=_.dtype
    )  # Create an identity matrix the size of the output

    @eqx.filter_jit
    def value_and_jacrev_x_flattened(x: jax.Array):
        """
        A helper function that computes the function value and its flattened Jacobian at a given point.

        Parameters
        ----------
        x : jax.Array
            A single input to function `f`.

        Returns
        -------
        jax.Array
            The function value concatenated with its flattened Jacobian vector.
        """
        y, pullback = vjp(f, x)
        return jnp.concatenate([y, vmap(pullback)(basis)[0].ravel()])

    return vmap(value_and_jacrev_x_flattened)(xs)


# def create_mean_h1_norm_l2_errors_and_norms(dM: int, batch_size: int) -> Callable:
#     """
#     Creates a function to compute mean L2 errors for both outputs and their Jacobians,
#     normalized mean squared errors, and returns these values along with the updated
#     batch index.

#     Parameters
#     ----------
#     dM : int
#         A multiplier for the Jacobian part of the loss, enhancing its impact relative
#         to the output's loss.
#     batch_size : int
#         The number of samples per batch.

#     Returns
#     -------
#     Callable
#         A function that computes the L2 loss and its mean for the given neural network
#         predictions and actual values, along with their Jacobians. It also computes
#         the normalized losses using actual L2 norms. This function returns the next batch
#         index and the computed metrics.

#     Notes
#     -----
#     The function produced by this factory is intended for use in batch processing
#     scenarios where sequential batches of data are processed through the model.
#     The provided function uses JAX operations to perform these computations efficiently.
#     The mean squared errors are computed for both the predicted outputs and their Jacobians.
#     """

#     def mean_h1_norm_l2_errors_and_norms(
#         nn: eqx.Module,
#         X: jax.Array,
#         Y: jax.Array,
#         dYdX: jax.Array,
#         Y_L2_norms: jax.Array,
#         dYdX_L2_norms: jax.Array,
#         end_idx: int,
#     ) -> Tuple[int, float, float, float, float]:
#         """
#         Computes the mean L2 losses for predictions and their Jacobians, along with
#         normalized values based on actual L2 norms.

#         Parameters
#         ----------
#         nn : eqx.Module
#             The neural network model that will provide predictions.
#         X : jax.Array
#             Input features for the neural network.
#         Y : jax.Array
#             Actual output values to compare against predictions.
#         dYdX : jax.Array
#             Actual Jacobians of the outputs.
#         Y_L2_norms : jax.Array
#             L2 norms of the actual output values, used for normalizing errors.
#         dYdX_L2_norms : jax.Array
#             L2 norms of the actual output Jacobians.
#         end_idx : int
#             The starting index of the current batch in the dataset.

#         Returns
#         -------
#         Tuple[int, float, float, float, float]
#             A tuple containing the new end index for the next batch, the mean squared
#             error for the outputs, the mean squared error for the Jacobians, and the
#             normalized errors for both the outputs and Jacobians, respectively.
#         """
#         predicted_Y, predicted_dYdX = __value_and_jacrev(
#             nn, jittable_slice(X, end_idx, batch_size)
#         )
#         mse_i = jnp.mean(
#             optax.l2_loss(predicted_Y.squeeze(), jittable_slice(Y, end_idx, batch_size)), axis=1
#         )
#         msje_i = jnp.mean(
#             optax.l2_loss(predicted_dYdX.squeeze(), jittable_slice(dYdX, end_idx, batch_size))
#             * dM,
#             axis=(1, 2),
#         )

#         return (
#             end_idx + batch_size,
#             jnp.mean(mse_i),
#             jnp.mean(msje_i),
#             np.mean(
#                 __elementwise_div(mse_i, jittable_slice(Y_L2_norms, end_idx, batch_size))
#             ),
#             jnp.mean(
#                 __elementwise_div(msje_i, jittable_slice(dYdX_L2_norms, end_idx, batch_size))
#             ),
#         )

#     return mean_h1_norm_l2_errors_and_norms


@eqx.filter_jit
def mean_h1_seminorm_and_l2_errors_and_rel_errors_flat(
    nn: eqx.Module,
    dY: int,
    X: jax.Array,
    Y_dYdX: jax.Array,
    Y_L2_norms: jax.Array,
    dYdX_L2_norms: jax.Array,
) -> jax.Array:
    """
    Computes mean squared errors for a flat structure of predicted and actual values
    and their Jacobians over a batch, and normalizes these errors by the L2 norms of
    the actual values and Jacobians.

    Parameters
    ----------
    nn : eqx.Module
        The neural network model that predicts both values and their Jacobians.
    dY : int
        The dimension marking the split between values and their Jacobians in the
        flattened array.
    X : jax.Array
        The input features for the neural network.
    Y_dYdX : jax.Array
        The combined flat array of actual output values and their Jacobians for comparison.
    Y_L2_norms : jax.Array
        The L2 norms of the actual output values.
    dYdX_L2_norms : jax.Array
        The L2 norms of the actual Jacobians.

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
    predicted_Y_dYdX = __value_and_jacrev_flattened(nn, X)
    Yhat_Y_L2_errors, dYhatdX_dYdX_L2_errors = tuple(
        map(
            lambda x: jnp.sum(x, axis=1),
            jnp.split(
                vmap(lambda x, y: (x - y) ** 2, (0, 0), 0)(
                    predicted_Y_dYdX.squeeze(), Y_dYdX
                ),
                [dY],
                axis=1,
            ),
        )
    )

    return jnp.array(
        [
            jnp.mean(Yhat_Y_L2_errors),
            jnp.mean(dYhatdX_dYdX_L2_errors),
            jnp.mean(__elementwise_div(Yhat_Y_L2_errors, Y_L2_norms)),
            jnp.mean(__elementwise_div(dYhatdX_dYdX_L2_errors, dYdX_L2_norms)),
        ]
    )


@eqx.filter_jit
def __batch_mean_h1_seminorm_and_l2_errors_and_rel_errors_flat(
    nn: eqx.Module,
    dY: int,
    batch_size: int,
    X: jax.Array,
    Y_dYdX: jax.Array,
    Y_L2_norms: jax.Array,
    dYdX_L2_norms: jax.Array,
    end_idx: int,
) -> jax.Array: 
    """
    Computes mean squared errors for a flat structure of predicted and actual values
    and their Jacobians over a batch, and normalizes these errors by the L2 norms of
    the actual values and Jacobians.

    Parameters
    ----------
    nn : eqx.Module
        The neural network model that predicts both values and their Jacobians.
    dY : int
        The dimension marking the split between values and their Jacobians in the
        flattened array.
    batch_size : int
        The number of samples per batch.
    X : jax.Array
        The input features for the neural network.
    Y_dYdX : jax.Array
        The combined flat array of actual output values and their Jacobians for comparison.
    Y_L2_norms : jax.Array
        The L2 norms of the actual output values.
    dYdX_L2_norms : jax.Array
        The L2 norms of the actual Jacobians.
    end_idx : int
        The starting index of the batch in the overall dataset.

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
    predicted_Y_dYdX = __value_and_jacrev_flattened(
        nn, jittable_slice(X, end_idx, batch_size)
    )
    Yhat_Y_L2_errors, dYhatdX_dYdX_L2_errors = tuple(
        map(
            lambda x: jnp.sum(x, axis=1),
            jnp.split(
                vmap(lambda x, y: (x - y) ** 2, (0, 0), 0)(
                    predicted_Y_dYdX.squeeze(),
                    jittable_slice(Y_dYdX, end_idx, batch_size),
                ),
                [dY],
                axis=1,
            ),
        )
    )

    return jnp.array(
        [
            jnp.mean(Yhat_Y_L2_errors),
            jnp.mean(dYhatdX_dYdX_L2_errors),
            jnp.mean(
                __elementwise_div(
                    Yhat_Y_L2_errors, jittable_slice(Y_L2_norms, end_idx, batch_size)
                )
            ),
            jnp.mean(
                __elementwise_div(
                    dYhatdX_dYdX_L2_errors,
                    jittable_slice(dYdX_L2_norms, end_idx, batch_size),
                )
            ),
        ]
    )


# @eqx.filter_jit
# def __batch_mean_h1_norm_l2_errors_and_norms(
#     nn: eqx.Module,
#     dM: int,
#     batch_size: int,
#     X: jax.Array,
#     Y: jax.Array,
#     dYdX: jax.Array,
#     Y_L2_norms: jax.Array,
#     dYdX_L2_norms: jax.Array,
#     end_idx: int,
# ) -> jax.Array:
#     """
#     Computes batch-based mean L2 norms of the differences between predicted and actual
#     values and their Jacobians, normalized by the actual norms, providing a measure
#     of prediction accuracy and sensitivity.

#     Parameters
#     ----------
#     nn : eqx.Module
#         The neural network model that predicts both values and their Jacobians.
#     dM : int
#         A scaling factor for adjusting the contribution of Jacobian errors.
#     batch_size : int
#         The number of samples per batch.
#     X : jax.Array
#         Input features for the neural network.
#     Y : jax.Array
#         Actual output values for comparison.
#     dYdX : jax.Array
#         Actual Jacobians of the outputs for comparison.
#     Y_L2_norms : jax.Array
#         L2 norms of the actual output values, used for normalizing errors.
#     dYdX_L2_norms : jax.Array
#         L2 norms of the actual output Jacobians, used for normalizing errors.
#     end_idx : int
#         The starting index of the batch in the overall dataset.

#     Returns
#     -------
#     jax.Array
#         An array containing the mean L2 norm of errors for the output predictions,
#         the mean L2 norm of errors for the Jacobian predictions, and their respective
#         normalized errors.

#     Notes
#     -----
#     This function is optimized for performance by using JAX's `vmap` and `jit` functions
#     to apply vectorized operations efficiently. The operations include calculating norms
#     of errors and normalizing these errors by the respective actual norms to account for
#     data magnitude and improve comparability.
#     """
#     predicted_Y, predicted_dYdX = __value_and_jacrev(nn, jittable_slice(X, end_idx, batch_size))
#     Yhat_Y_L2_errors, dYhatdX_dYdX_L2_errors = vmap(jax.jit(lambda x: jnp.linalg.norm(x) ** 2))(
#         predicted_Y.squeeze() - jittable_slice(Y, end_idx, batch_size)
#     ), vmap(jax.jit(lambda x: jnp.linalg.norm(x) ** 2))(
#         predicted_dYdX.squeeze() - jittable_slice(dYdX, end_idx, batch_size)
#     )

#     return jnp.array(
#         [
#             jnp.mean(Yhat_Y_L2_errors),
#             jnp.mean(dYhatdX_dYdX_L2_errors),
#             jnp.mean(
#                 __elementwise_div(
#                     Yhat_Y_L2_errors, jittable_slice(Y_L2_norms, end_idx, batch_size)
#                 )
#             ),
#             jnp.mean(
#                 __elementwise_div(
#                     dYhatdX_dYdX_L2_errors, jittable_slice(dYdX_L2_norms, end_idx, batch_size)
#                 )
#             ),
#         ]
#     )


@eqx.filter_jit
def mean_h1_seminorm_and_l2_errors_and_rel_errors_flat(
    nn: eqx.Module,
    dY: int,
    X: jax.Array,
    Y_dYdX: jax.Array,
    Y_L2_norms: jax.Array,
    dYdX_L2_norms: jax.Array,
) -> jax.Array:
    """
    Computes mean squared errors for a flat structure of predicted and actual values
    and their Jacobians over a batch, and normalizes these errors by the L2 norms of
    the actual values and Jacobians.

    Parameters
    ----------
    nn : eqx.Module
        The neural network model that predicts both values and their Jacobians.
    dY : int
        The dimension marking the split between values and their Jacobians in the
        flattened array.
    X : jax.Array
        The input features for the neural network.
    Y_dYdX : jax.Array
        The combined flat array of actual output values and their Jacobians for comparison.
    Y_L2_norms : jax.Array
        The L2 norms of the actual output values.
    dYdX_L2_norms : jax.Array
        The L2 norms of the actual Jacobians.

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
    predicted_Y_dYdX = __value_and_jacrev_flattened(nn, X)
    Yhat_Y_L2_errors, dYhatdX_dYdX_L2_errors = tuple(
        map(
            lambda x: jnp.sum(x, axis=1),
            jnp.split(
                vmap(lambda x, y: (x - y) ** 2, (0, 0), 0)(
                    predicted_Y_dYdX.squeeze(), Y_dYdX
                ),
                [dY],
                axis=1,
            ),
        )
    )

    return jnp.array(
        [
            jnp.mean(Yhat_Y_L2_errors),
            jnp.mean(dYhatdX_dYdX_L2_errors),
            jnp.mean(__elementwise_div(Yhat_Y_L2_errors, Y_L2_norms)),
            jnp.mean(__elementwise_div(dYhatdX_dYdX_L2_errors, dYdX_L2_norms)),
        ]
    )


@eqx.filter_jit
def mean_l2_errors_and_rel_errors(
    nn: eqx.Module,
    X: jax.Array,
    Y: jax.Array,
    Y_L2_norms: jax.Array,
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
        The input features for the neural network.
    Y : jax.Array
        The actual output values for error calculation.
    Y_L2_norms : jax.Array
        The L2 norms of the actual output values.
    dYdX_L2_norms : jax.Array
        The L2 norms of the actual Jacobians.

    Returns
    -------
    jax.Array
        An array containing the mean squared errors for the outputs and Jacobians,
        as well as their respective normalized mean squared errors, calculated as:
        [mean MSE for Y,  normalized MSE for Y].

    Notes
    -----
    This function leverages JAX's `vmap` to vectorize the computation of squared differences,
    applies a lambda function to compute squared errors, and splits the result to handle
    outputs and Jacobians separately. This ensures efficient batch processing and is
    particularly useful for gradient-based optimization where sensitivity information is crucial.
    """
    predicted_Y_dYdX = __value_and_jacrev_flattened(nn, X)
    Yhat_Y_L2_errors = (jnp.sum((predicted_Y_dYdX.squeeze() - Y) ** 2.0, axis=1),)
    return jnp.array(
        [
            jnp.mean(Yhat_Y_L2_errors),
            jnp.mean(__elementwise_div(Yhat_Y_L2_errors, Y_L2_norms)),
        ]
    )


@eqx.filter_jit
def __batch_mean_l2_errors_and_rel_errors(
    nn: eqx.Module,
    batch_size: int,
    X: jax.Array,
    Y: jax.Array,
    Y_L2_norms: jax.Array,
    end_idx: int,
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    """
    Computes mean L2 errors and their normalized versions for a batch of data
    using a vectorized neural network model.

    Parameters
    ----------
    nn : eqx.Module
        The neural network model which will process the input data.
    batch_size : int
        The number of samples per batch.
    X : jax.Array
        The input features for the neural network.
    Y : jax.Array
        The actual output values against which predictions are compared.
    Y_L2_norms : jax.Array
        The L2 norms of the actual output values, used to normalize the mean squared errors.
    end_idx : int
        The index of the first sample in the current batch within the overall dataset.

    Returns
    -------
    Tuple[int, jnp.ndarray, jnp.ndarray]
        A tuple containing:
        - The updated end index after processing one batch.
        - The mean L2 error for the batch, scaled by the reciprocal of the batch size.
        - The mean normalized L2 error for the batch, also scaled by the reciprocal of the batch size.

    Notes
    -----
    The function utilizes JAX's `vmap` to apply the neural network model in a vectorized
    manner across the batch. This allows for efficient batch processing. The L2 loss is
    computed using Optax's `l2_loss` function, and results are averaged across the batch.
    The normalization of errors uses elementwise division to adjust the errors relative to
    the magnitude of the actual outputs, providing a measure of error relative to the scale
    of the data.
    """
    predicted_Y = vmap(nn)(jittable_slice(X, end_idx, batch_size))
    Yhat_Y_L2_errors = (
        jnp.sum(
            (predicted_Y.squeeze() - jittable_slice(Y, end_idx, batch_size)) ** 2.0,
            axis=1,
        ),
    )

    return (
        end_idx + batch_size,
        jnp.mean(Yhat_Y_L2_errors),
        jnp.mean(
            __elementwise_div(
                Yhat_Y_L2_errors, jittable_slice(Y_L2_norms, end_idx, batch_size)
            )
        ),
    )


# def create_mean_l2_errors_and_norms(batch_size: int) -> Callable:
#     """
#     Creates a function that computes the mean L2 loss and normalized L2 loss for
#     batches of predictions from a neural network, relative to actual output values.

#     Parameters
#     ----------
#     batch_size : int
#         The number of samples per batch used in the computation.

#     Returns
#     -------
#     Callable
#         A function that computes mean L2 errors and normalized L2 errors for a
#         given batch of data when provided with a neural network model, input features,
#         actual outputs, and L2 norms of actual outputs. The function also updates the
#         index of the end of the batch in the dataset.

#     Notes
#     -----
#     The function returned by this factory is jitted for performance using Equinox's
#     `filter_jit`. It uses `vmap` to vectorize model prediction over the batch for
#     efficiency. The function computes:
#     - The mean squared error for each sample in the batch.
#     - The mean of these errors normalized by the actual L2 norms of the outputs.
#     Each output is scaled by the reciprocal of the batch size to normalize across
#     different batch sizes, which helps in averaging the results correctly in further aggregations.
#     """

#     one_over_n_batches = 1.0 / batch_size

#     @eqx.filter_jit
#     def __mean_l2_errors_and_norms(
#         nn: eqx.Module, X: jax.Array, Y: jax.Array, Y_L2_norms: jax.Array, end_idx: int
#     ):
#         """
#         Computes mean L2 errors and normalized L2 errors for a batch.

#         Parameters
#         ----------
#         nn : eqx.Module
#             The neural network model that predicts outputs from inputs.
#         X : jax.Array
#             The array of input features.
#         Y : jax.Array
#             The array of actual output values to compare against predictions.
#         Y_L2_norms : jax.Array
#             The L2 norms of the actual output values, used for normalizing errors.
#         end_idx : int
#             The current starting index of the batch in the overall dataset.

#         Returns
#         -------
#         tuple
#             A tuple containing:
#             - The updated end index after processing one batch.
#             - The mean L2 error for the batch, scaled by the reciprocal of the number of batches.
#             - The mean normalized L2 error for the batch, also scaled accordingly.
#         """
#         predicted_Y = vmap(nn)(jittable_slice(X, end_idx, batch_size))
#         mse_i = jnp.mean(
#             optax.l2_loss(predicted_Y.squeeze(), jittable_slice(Y, end_idx, batch_size)), axis=1
#         )

#         return (
#             end_idx + batch_size,
#             one_over_n_batches * jnp.mean(mse_i),
#             one_over_n_batches
#             * jnp.mean(
#                 __elementwise_div(mse_i, jittable_slice(Y_L2_norms, end_idx, batch_size))
#             ),
#         )

#     return __mean_l2_errors_and_norms


# @eqx.filter_jit
# def __mean_h1_norm_loss(
#     dM: int,
#     nn: eqx.Module,
#     input_X: jax.Array,
#     actual_Y: jax.Array,
#     actual_dYdX: jax.Array,
# ) -> jnp.ndarray:
#     """
#     Computes the weighted mean H1 norm loss for a neural network model based on
#     the given inputs and their Jacobians. The loss combines the L2 norm losses of the
#     outputs and the Jacobians, with the latter scaled by a factor dependent on the
#     dimensionality of the model's output.

#     Parameters
#     ----------
#     dM : int
#         The dimensionality multiplier for the Jacobian's contribution to the total loss.
#         This scalar amplifies the importance of the Jacobian's accuracy in the total loss calculation.
#     nn : eqx.Module
#         The neural network model which predicts both values and their Jacobians.
#     input_X : jax.Array
#         The input data to the neural network model.
#     actual_Y : jax.Array
#         The actual output values against which the model's predictions are compared.
#     actual_dYdX : jax.Array
#         The actual Jacobians of the outputs against which the model's predicted Jacobians are compared.

#     Returns
#     -------
#     jnp.ndarray
#         A scalar JAX array representing the total mean H1 norm loss, calculated by summing the
#         mean L2 loss of the outputs and the scaled mean L2 loss of the Jacobians.

#     Notes
#     -----
#     This function is particularly useful for applications where the accuracy of both the output values
#     and their gradients (Jacobians) are crucial, such as in gradient-based optimization problems or
#     when using physics-informed neural networks.
#     """
#     predicted_Y, predicted_dYdX = __value_and_jacrev(nn, input_X)
#     return (
#         jnp.mean(optax.l2_loss(predicted_Y.squeeze(), actual_Y))
#         + jnp.mean(optax.l2_loss(predicted_dYdX.squeeze(), actual_dYdX)) * dM * 10
#     )


# def squared_error(a, b):


@eqx.filter_jit
def __mean_h1_norm_loss_flattened(
    nn: eqx.Module, input_X: jax.Array, actual_Y_dYdX: jax.Array
) -> jnp.ndarray:
    """
    Computes the mean L2 norm loss between flattened predicted and actual values
    of both outputs and their Jacobians from a neural network model.

    Parameters
    ----------
    nn : eqx.Module
        The neural network model used to compute predictions. This model should be capable
        of outputting both values and their Jacobians, likely through a custom forward
        and Jacobian reverse function.
    input_X : jax.Array
        Input features array to the neural network, from which predictions are generated.
    actual_Y_dYdX : jax.Array
        Actual combined output and Jacobian values array for comparison, typically
        flattened to match the predicted format.

    Returns
    -------
    jnp.ndarray
        A scalar JAX array representing the mean L2 norm loss across all predicted and actual
        values, providing a measure of prediction accuracy for both the values and their
        Jacobians.

    Notes
    -----
    This function is particularly useful in scenarios where both the values and their
    Jacobians are significant, such as in physics-informed neural networks or when
    gradient information is crucial. It uses a specialized version of the model's forward
    pass (`__value_and_jacrev_flattened`) to generate these combined predictions efficiently.
    """
    predicted_Y_dYdX = __value_and_jacrev_flattened(nn, input_X)
    return jnp.mean(optax.l2_loss(predicted_Y_dYdX.squeeze(), actual_Y_dYdX))


# def create_mean_h1_norm_loss(dM: int) -> Callable:
#     """
#     Creates a function that computes the H1 norm loss for predictions and their
#     Jacobians from a neural network model, with an emphasis on the Jacobians' importance
#     weighted by a factor.

#     Parameters
#     ----------
#     dM : int
#         A scalar that amplifies the contribution of the Jacobians' loss in the total loss,
#         effectively weighting the importance of the Jacobians in the H1 norm calculation.

#     Returns
#     -------
#     Callable
#         A function that when given a neural network, input features, actual outputs,
#         and actual Jacobians, returns the computed H1 norm loss.
#         This loss combines the L2 losses of both outputs and Jacobians, with the
#         Jacobians' losses being scaled by `dM * 5`.

#     Notes
#     -----
#     The `eqx.filter_jit` decorator is used to JIT compile the function, enhancing performance
#     by ensuring that the function computations are optimized and executed efficiently on
#     compatible hardware accelerators like GPUs or TPUs.

#     The created function computes the loss by:
#     - Predicting outputs and their Jacobians from the input features using the neural network model.
#     - Calculating the L2 loss between the predicted outputs and the actual outputs, as well as
#       between the predicted Jacobians and the actual Jacobians.
#     - The loss from the Jacobians is given more weight in the loss calculation, recognizing
#       their importance in applications where Jacobians (sensitivity or gradient information)
#       are crucial.
#     """

#     @eqx.filter_jit
#     def mean_h1_norm_loss(
#         nn: eqx.Module, input_X: jax.Array, actual_Y: jax.Array, actual_dYdX: jax.Array
#     ) -> jnp.ndarray:
#         """
#         Computes the combined H1 norm loss.

#         Parameters
#         ----------
#         nn : eqx.Module
#             The neural network model to compute predictions from.
#         input_X : jax.Array
#             Input array to the neural network.
#         actual_Y : jax.Array
#             Actual output values for comparison.
#         actual_dYdX : jax.Array
#             Actual Jacobian values for comparison.

#         Returns
#         -------
#         jnp.ndarray
#             A scalar JAX array representing the combined H1 norm loss, where the loss
#             from the Jacobians is scaled up significantly.
#         """
#         predicted_Y, predicted_dYdX = __value_and_jacrev(nn, input_X)
#         return (
#             jnp.mean(optax.l2_loss(predicted_Y.squeeze(), actual_Y))
#             + jnp.mean(optax.l2_loss(predicted_dYdX.squeeze(), actual_dYdX)) * dM * 5
#         )

#     return mean_h1_norm_loss


# def create_mean_h1_norm_loss_more_correct(dM: int) -> Callable:
#     """
#     Creates a function that computes a combined mean H1 norm loss for predictions
#     and their Jacobians, given a neural network model.

#     Parameters
#     ----------
#     dM : int
#         The dimensionality of the model's output used for scaling the Jacobian loss.

#     Returns
#     -------
#     Callable
#         A function that, when called with a neural network, input features, actual outputs,
#         and actual Jacobians, computes the mean H1 norm loss.

#     Notes
#     -----
#     The resulting function computes the loss by:
#     - Predicting outputs and their Jacobians from the input features using the provided
#       neural network model.
#     - Calculating the L2 loss between the predicted and actual outputs, and similarly for
#       the predicted and actual Jacobians.
#     - Summing these losses, with the Jacobian loss scaled by the model's output dimensionality,
#       then averaging over samples to provide a mean value.

#     The use of `eqx.filter_jit` optimizes the created function for faster execution by JIT compiling it.
#     """

#     @eqx.filter_jit
#     def mean_h1_norm_loss(
#         nn: eqx.Module, input_X: jax.Array, actual_Y: jax.Array, actual_dYdX: jax.Array
#     ) -> jnp.ndarray:
#         """
#         Computes the mean H1 norm loss.

#         Parameters
#         ----------
#         nn : eqx.Module
#             The neural network model to compute predictions from.
#         input_X : jax.Array
#             Input array to the neural network.
#         actual_Y : jax.Array
#             Actual output values for comparison.
#         actual_dYdX : jax.Array
#             Actual Jacobian values for comparison.

#         Returns
#         -------
#         jnp.ndarray
#             A scalar JAX array representing the mean H1 norm loss.
#         """
#         predicted_Y, predicted_dYdX = __value_and_jacrev(nn, input_X)
#         return (
#             jnp.mean(jnp.sum(optax.l2_loss(predicted_Y.squeeze(), actual_Y), axis=1))
#             + jnp.mean(
#                 jnp.sum(
#                     optax.l2_loss(predicted_dYdX.squeeze(), actual_dYdX), axis=(1, 2)
#                 )
#             )
#             * dM
#         )

#     return mean_h1_norm_loss


@eqx.filter_jit
def __mean_l2_norm_loss(
    nn: eqx.Module, input_X: jax.Array, actual_Y: jax.Array
) -> float:
    """
    Computes the MSE (mean L2 norm) loss for predictions made by the neural network model.

    Parameters
    ----------
    nn : eqx.Module
        An Equinox neural network model that can process inputs and produce outputs.
    input_X : jax.Array
        Input array to the neural network, which will be processed to generate predictions.
    actual_Y : jax.Array
        The actual values for comparison, used to compute the L2 loss against predictions.

    Returns
    -------
    float
        The mean L2 norm loss between the predicted outputs and the actual outputs.

    Notes
    -----
    This function applies the model to the input data, squeezes the output dimensions,
    and then computes the mean L2 loss using Optax's `l2_loss` function.
    """
    return jnp.mean(optax.l2_loss(nn(input_X).squeeze(), actual_Y))


# Gradient functions using Equinox's filtering on gradient calculation
# __grad_mean_h1_norm_loss = eqx.filter_grad(__mean_h1_norm_loss)
grad_mean_l2_norm_loss = eqx.filter_grad(__mean_l2_norm_loss)

# Lambda to create gradient function for mean H1 norm loss dynamically based on dimensionality
# __grad_mean_l2_norm_loss_flattened = eqx.filter_grad(__mean_l2_norm_loss)

# Gradient function for the flattened H1 norm loss
grad_mean_h1_norm_loss_flattened = eqx.filter_grad(__mean_h1_norm_loss_flattened)


@jax.jit
def __elementwise_div(
    scores: Union[np.ndarray, jax.Array], normalizers: Union[np.ndarray, jax.Array]
) -> Union[np.ndarray, jax.Array]:
    """
    Performs element-wise division of two arrays.

    Parameters
    ----------
    scores : Union[np.ndarray, jax.Array]
        An array of numerators. This could be a NumPy array or a JAX array, depending on the context
        in which the function is used.
    normalizers : Union[np.ndarray, jax.Array]
        An array of denominators, with the same dimensions as the `scores` array.

    Returns
    -------
    Union[np.ndarray, jax.Array]
        The result of element-wise division of `scores` by `normalizers`. The type of the output array
        will match the type of the input arrays.

    Notes
    -----
    This function is marked as internal (prefixed with double underscores) suggesting that it is
    intended for internal use within the module or package. This function handles division directly,
    which may include broadcasting if `scores` and `normalizers` have compatible shapes but not
    exactly the same shape.
    """
    return scores / normalizers


def compute_flattened_h1_loss_metrics(
    nn: eqx.Module,
    dY: int,
    X: jax.Array,  # Input features are JAX arrays
    Y_dYdX: jax.Array,  # Combined output and Jacobian data are JAX arrays
    Y_L2_norms: jax.Array,
    dYdX_L2_norms: jax.Array,
) -> Tuple[float, float, float, float]:
    """
    Computes H1 loss metrics for a neural network over multiple batches, assuming
    a flat data structure for outputs and Jacobians. Although the function uses
    JAX arrays for handling data to leverage JAX's capabilities, computation within
    the function utilizes NumPy for non-GPU operations to ensure 64-bit accuracy.

    Parameters
    ----------
    nn : Any
        The neural network model used for predictions.
    dY : int
        The dimensionality of the model's output, used to split Y_dYdX.
    X : jax.Array
        The input features array.
    Y_dYdX : jax.Array
        The combined array of outputs and their Jacobians.
    Y_L2_norms : jax.Array
        The L2 norms of the outputs.
    dYdX_L2_norms : jax.Array
        The L2 norms of the Jacobians.

    Returns
    -------
    Tuple[float, float, float, float]
        - Accuracy % based on L2 norm for the output predictions.
        - Accuracy % based on H1 norm for the Jacobian predictions.
        - Mean squared error of the outputs.
        - Mean squared error of the Jacobians.

    Notes
    -----
    The function uses JAX arrays for data input and output but processes calculations with NumPy to avoid unnecessary use of JAX's GPU acceleration for operations that do not require it. This approach ensures compatibility and efficiency for specific computational contexts.
    """
    errors = mean_h1_seminorm_and_l2_errors_and_rel_errors_flat(
        nn, dY, X, Y_dYdX, Y_L2_norms, dYdX_L2_norms
    )
    return *(100 * (1.0 - np.sqrt(errors[2:]))), *(errors[0:2])


def compute_l2_loss_metrics(
    nn: eqx.Module,
    X: jax.Array,  # Input features are JAX arrays
    Y: jax.Array,  # Combined output and Jacobian data are JAX arrays
    Y_L2_norms: jax.Array,
) -> Tuple[float, float, float, float]:
    """
    Computes H1 loss metrics for a neural network over multiple batches, assuming
    a flat data structure for outputs and Jacobians. Although the function uses
    JAX arrays for handling data to leverage JAX's capabilities, computation within
    the function utilizes NumPy for non-GPU operations to ensure 64-bit accuracy.

    Parameters
    ----------
    nn : Any
        The neural network model used for predictions.
    X : jax.Array
        The input features array.
    Y : jax.Array
        The outputs.
    Y_L2_norms : jax.Array
        The L2 norms of the outputs.
    dYdX_L2_norms : jax.Array
        The L2 norms of the Jacobians.

    Returns
    -------
    Tuple[float, float]
        - Accuracy % based on L2 norm for the output predictions.
        - Mean squared error of the outputs.

    Notes
    -----
    The function uses JAX arrays for data input and output but processes calculations with NumPy
    to avoid unnecessary use of JAX's GPU acceleration for operations that do not require it.
    This approach ensures compatibility and efficiency for specific computational contexts.
    """
    errors = mean_l2_errors_and_rel_errors(nn, X, Y, Y_L2_norms)
    return 100 * (1.0 - np.sqrt(errors[1])), errors[0]


def compute_batched_flattened_h1_loss_metrics(
    nn: eqx.Module,
    dY: int,
    batch_size: int,
    n_batches: int,
    X: jax.Array,  # Input features are JAX arrays
    Y_dYdX: jax.Array,  # Combined output and Jacobian data are JAX arrays
    Y_L2_norms: jax.Array,
    dYdX_L2_norms: jax.Array,
) -> Tuple[float, float, float, float]:
    """
    Computes H1 loss metrics for a neural network over multiple batches, assuming
    a flat data structure for outputs and Jacobians. Although the function uses
    JAX arrays for handling data to leverage JAX's capabilities, computation within
    the function utilizes NumPy for non-GPU operations to ensure 64-bit accuracy.

    Parameters
    ----------
    nn : Any
        The neural network model used for predictions.
    dY : int
        The dimensionality of the model's output, used in loss scaling.
    batch_size : int
        The number of samples in each batch.
    n_batches : int
        The total number of batches to process.
    X : jax.Array
        The input features array.
    Y_dYdX : jax.Array
        The combined array of outputs and their Jacobians.
    Y_L2_norms : jax.Array
        The L2 norms of the outputs.
    dYdX_L2_norms : jax.Array
        The L2 norms of the Jacobians.

    Returns
    -------
    Tuple[float, float, float, float]
        - Accuracy % based on L2 norm for the output predictions.
        - Accuracy % based on H1 norm for the Jacobian predictions.
        - Mean squared error of the outputs.
        - Mean squared error of the Jacobians.

    Notes
    -----
    The function uses JAX arrays for data input and output but processes calculations with NumPy to avoid unnecessary use of JAX's GPU acceleration for operations that do not require it. This approach ensures compatibility and efficiency for specific computational contexts.
    """
    end_idx, errors = 0, np.zeros((4,), dtype=np.float64)
    for _ in range(n_batches):
        errors += __batch_mean_h1_seminorm_and_l2_errors_and_rel_errors_flat(
            nn, dY, batch_size, X, Y_dYdX, Y_L2_norms, dYdX_L2_norms, end_idx
        )
        end_idx += batch_size
    errors /= n_batches
    return *(100 * (1.0 - np.sqrt(errors[2:]))), *(errors[0:2])


def compute_batched_l2_loss_metrics(
    nn: eqx.Module,
    batch_size: int,
    n_batches: int,
    X: jax.Array,  # Input features are JAX arrays
    Y: jax.Array,  # Output are JAX arrays
    Y_L2_norms: jax.Array,
) -> Tuple[float, float]:
    """
    Computes L2 loss metrics for a neural network over multiple batches, assuming
    a flat data structure for outputs. Although the function uses
    JAX arrays for handling data to leverage JAX's capabilities, computation within
    the function utilizes NumPy for non-GPU operations to ensure 64-bit accuracy.

    Parameters
    ----------
    nn : Any
        The neural network model used for predictions.
    dY : int
        The dimensionality of the model's output, used in loss scaling.
    batch_size : int
        The number of samples in each batch.
    n_batches : int
        The total number of batches to process.
    X : jax.Array
        The input features array.
    Y : jax.Array
        The array of outputs.
    Y_L2_norms : jax.Array
        The L2 norms of the outputs.

    Returns
    -------
    Tuple[float, float]
        - Accuracy % based on L2 norm for the output predictions.
        - Mean squared error of the outputs.

    Notes
    -----
    The function uses JAX arrays for data input and output but processes calculations
    with NumPy to avoid unnecessary use of JAX's GPU acceleration for operations that
    do not require it. This approach ensures compatibility and efficiency for specific
    computational contexts.
    """
    end_idx, errors = 0, np.zeros((4,), dtype=np.float64)
    for _ in range(n_batches):
        errors += __batch_mean_l2_errors_and_rel_errors(
            nn, batch_size, X, Y, Y_L2_norms, end_idx
        )
        end_idx += batch_size
    errors /= n_batches
    return *(100 * (1.0 - np.sqrt(errors[2:]))), *(errors[0:2])


# def compute_h1_loss_metrics(
#     nn: eqx.Module,
#     dY: int,
#     batch_size: int,
#     n_batches: int,
#     X: jax.Array,  # Input features are JAX arrays
#     Y_dYdX: jax.Array,  # Combined output and Jacobian data are JAX arrays
#     Y_L2_norms: jax.Array,
#     dYdX_L2_norms: jax.Array,
# ) -> Tuple[float, float, float, float]:
#     """
#     Computes H1 loss metrics for a neural network over multiple batches, assuming
#     a flat data structure for outputs and Jacobians. Although the function uses
#     JAX arrays for handling data to leverage JAX's capabilities, computation within
#     the function utilizes NumPy for non-GPU operations.

#     Parameters
#     ----------
#     nn : Any
#         The neural network model used for predictions.
#     dY : int
#         The dimensionality of the model's output, used in loss scaling.
#     batch_size : int
#         The number of samples in each batch.
#     n_batches : int
#         The total number of batches to process.
#     X : jax.Array
#         The input features array.
#     Y_dYdX : jax.Array
#         The combined array of outputs and their Jacobians.
#     Y_L2_norms : jax.Array
#         The L2 norms of the outputs.
#     dYdX_L2_norms : jax.Array
#         The L2 norms of the Jacobians.

#     Returns
#     -------
#     Tuple[float, float, float, float]
#         - Accuracy based on L2 norm for the output predictions.
#         - Accuracy based on H1 norm for the Jacobian predictions.
#         - Mean squared error of the outputs.
#         - Mean squared error of the Jacobians.

#     Notes
#     -----
#     The function uses JAX arrays for data input and output but processes calculations with NumPy to avoid unnecessary use of JAX's GPU acceleration for operations that do not require it. This approach ensures compatibility and efficiency for specific computational contexts.
#     """
#     end_idx, errors = 0, np.zeros((4,), dtype=np.float64)
#     for _ in range(n_batches):
#         errors += batch_mean_h1_norm_l2_errors_and_norms(
#             nn, dM, batch_size, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms, end_idx
#         )
#         end_idx += batch_size
#     errors /= n_batches
#     acc_l2, acc_h1 = 1.0 - np.sqrt(errors[2:])
#     return acc_l2, acc_h1, np.sum(errors[:2])


# # @eqx.filter_jit
# def batched_compute_h1_loss_metrics(
#     nn: eqx.Module,
#     dM: int,
#     batch_size: int,
#     n_batches: int,
#     X: jax.Array,
#     Y: jax.Array,
#     dYdX: jax.Array,
#     Y_L2_norms: jax.Array,
#     dYdX_L2_norms: jax.Array,
# ) -> tuple:
#     """
#     Computes H1 loss metrics in a batched manner for a given neural network and dataset.

#     Parameters
#     ----------
#     nn : eqx.Module
#         The neural network model used for predictions.
#     dM : int
#         The dimensionality of the model's input, used to adjust the loss scaling.
#     batch_size : int
#         The number of samples per batch.
#     n_batches : int
#         The total number of batches to process.
#     X : jax.Array
#         The input features array.
#     Y : jax.Array
#         The target output values array.
#     dYdX : jax.Array
#         The target Jacobians array.
#     Y_L2_norms : jax.Array
#         The L2 norms of the target outputs.
#     dYdX_L2_norms : jax.Array
#         The L2 norms of the target Jacobians.

#     Returns
#     -------
#     tuple
#         A tuple containing:
#         - L2 accuracy for the output predictions,
#         - H1 accuracy for the Jacobian predictions,
#         - Combined mean H1 norm loss across all batches.

#     Notes
#     -----
#     This function performs batched computations of mean squared errors and Jacobian errors
#     for a neural network using JAX's `lax.scan` for efficient processing. It computes the
#     error metrics per batch and aggregates them to calculate overall metrics.
#     """
#     nn_arr_part, nn_static_part = eqx.partition(nn, eqx.is_inexact_array)
#     one_over_n_batches = 1.0 / n_batches

#     def __batched_mean_h1_norm_l2_errors_and_norms(carry, end_idx):
#         nn_arr_part, partials = carry
#         nn = eqx.combine(nn_arr_part, nn_static_part)
#         predicted_Y, predicted_dYdX = __value_and_jacrev(
#             nn, jittable_slice(X, end_idx, batch_size)
#         )
#         mse_i, msje_i = jnp.mean(
#             optax.l2_loss(predicted_Y.squeeze(), jittable_slice(Y, end_idx, batch_size)), axis=1
#         ), jnp.mean(
#             optax.l2_loss(predicted_dYdX.squeeze(), jittable_slice(dYdX, end_idx, batch_size))
#             * dM,
#             axis=(1, 2),
#         )
#         nn_arr_part, _ = eqx.partition(nn, eqx.is_inexact_array)
#         partials = partials.at[0].add(jnp.mean(mse_i))
#         partials = partials.at[1].add(jnp.mean(msje_i))
#         partials = partials.at[2].add(
#             jnp.mean(
#                 __elementwise_div(mse_i, jittable_slice(Y_L2_norms, end_idx, batch_size))
#             )
#         )
#         partials = partials.at[3].add(
#             jnp.mean(
#                 __elementwise_div(msje_i, jittable_slice(dYdX_L2_norms, end_idx, batch_size))
#             )
#         )

#         return (nn_arr_part, partials), [0]

#     results = jax.lax.scan(
#         __batched_mean_h1_norm_l2_errors_and_norms,
#         (nn_arr_part, jnp.array([0.0, 0.0, 0.0, 0.0])),
#         jnp.arange(0, n_batches * batch_size, batch_size, dtype=jnp.uint32),
#     )[0][1]
#     return (
#         1.0 - jnp.sqrt(results.at[2].get() * one_over_n_batches),
#         1.0 - jnp.sqrt(results.at[3].get() * one_over_n_batches),
#         (results.at[0].get() + results.at[1].get()) * one_over_n_batches,
#     )


# def create_compute_h1_loss_metrics(dM: int, batch_size: int) -> Callable:
#     """
#     Creates a closure function that computes various loss metrics, including mean squared error (MSE),
#     mean squared Jacobian error (MSJE), and their relative errors, as well as accuracy measures for both
#     L2 and H1 norms.

#     Parameters
#     ----------
#     dM : int
#         Dimensionality of the model's input space, used in the H1 norm calculations.
#     batch_size : int
#         The number of samples in each batch used during the training or evaluation.

#     Returns
#     -------
#     Callable
#         A function that calculates loss metrics and accuracies for a given neural network,
#         input features (X), true output values (Y), Jacobians (dYdX), and their respective L2 norms,
#         across a specified number of batches.

#     Notes
#     -----
#     The returned function computes:
#     - Mean squared error (MSE) for the output values.
#     - Mean squared Jacobian error (MSJE) for the Jacobians.
#     - Relative mean squared errors for both the outputs and Jacobians.
#     - Accuracy for L2 and H1 norms, derived from the relative errors.

#     The calculations are performed by aggregating the errors across all batches and then computing
#     the average or total as appropriate. It uses an underlying function to calculate the mean errors
#     and norms per batch, which must be defined externally.
#     """
#     mean_h1_norm_errors_and_norms = create_mean_h1_norm_l2_errors_and_norms(
#         dM, batch_size
#     )

#     def compute_loss_metrics(nn, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms, n_batches):
#         mse = 0.0
#         msje = 0.0
#         rel_mse = 0.0
#         rel_msje = 0.0
#         end_idx = 0

#         for _ in range(n_batches):
#             end_idx, a, b, c, d = mean_h1_norm_errors_and_norms(
#                 nn, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms, end_idx
#             )
#             mse += a
#             msje += b
#             rel_mse += c
#             rel_msje += d

#         acc_l2 = 1.0 - jnp.sqrt(rel_mse / n_batches)
#         acc_h1 = 1.0 - jnp.sqrt(rel_msje / n_batches)
#         mean_h1_norm_loss = (mse + msje) / n_batches

#         return acc_l2, acc_h1, mean_h1_norm_loss

#     return compute_loss_metrics
