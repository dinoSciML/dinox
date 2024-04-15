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

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import vjp, vmap
from jax.lax import dynamic_slice_in_dim as dslice

__all__ = [  #    "compute_l2_loss_metrics",
    "compute_h1_loss_metrics",
    "batched_compute_h1_loss_metrics",
    "create_compute_h1_loss_metrics",
    "take_h1_step",
    "take_l2_step",
]


# @eqx.filter_jit
def __value_and_jacrev(f, xs):
    # No side effects
    _, pullback = vjp(f, xs[0])
    basis = jnp.eye(_.size, dtype=_.dtype)

    # @eqx.filter_jit
    def value_and_jacrev_x(x):
        y, pullback = vjp(f, x)
        jac = vmap(pullback)(basis)
        return y, jac[0]  # There is only one jacobian matrix here, so we extract it

    return vmap(value_and_jacrev_x)(xs)


@eqx.filter_jit
def __value_and_jacrev_flattened(f, xs):
    # No side effects
    _, pullback = vjp(f, xs[0])
    basis = jnp.eye(_.size, dtype=_.dtype)

    # total_len = _.shape*(1+xs[0].size)
    @eqx.filter_jit
    def value_and_jacrev_x_flattened(x):
        y, pullback = vjp(f, x)
        return jnp.concatenate([y, vmap(pullback)(basis)[0].ravel()])

    return vmap(value_and_jacrev_x_flattened)(xs)


def create_mean_h1_seminorm_l2_errors_and_norms(dM, batch_size):
    # No side effects

    # @eqx.filter_jit #end_idx should not be static
    def mean_h1_seminorm_l2_errors_and_norms(
        nn, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms, end_idx
    ):  # dM, batch_size
        predicted_Y, predicted_dYdX = __value_and_jacrev(
            nn, dslice(X, end_idx, batch_size)
        )
        mse_i, msje_i = jnp.mean(
            optax.l2_loss(predicted_Y.squeeze(), dslice(Y, end_idx, batch_size)),
            axis=1,
        ), jnp.mean(
            optax.l2_loss(
                predicted_dYdX.squeeze(),
                dslice(dYdX, end_idx, batch_size),
            )
            * dM,
            axis=(1, 2),
        )
        return (
            end_idx + batch_size,
            jnp.mean(mse_i),
            jnp.mean(msje_i),
            np.mean(__normalize_values(mse_i, dslice(Y_L2_norms, end_idx, batch_size))),
            jnp.mean(
                __normalize_values(msje_i, dslice(dYdX_L2_norms, end_idx, batch_size))
            ),
        )

    return mean_h1_seminorm_l2_errors_and_norms


@eqx.filter_jit
def batch_mean_h1_seminorm_l2_errors_and_norms_flat(
    nn, dY, batch_size, X, Y_dYdX, Y_L2_norms, dYdX_L2_norms, end_idx
):

    predicted_Y_dYdX = __value_and_jacrev_flattened(nn, dslice(X, end_idx, batch_size))
    Yhat_L2_norms, dYhatdX_L2_norms = tuple(
        map(
            lambda x: jnp.sum(x, axis=1),
            jnp.split(
                vmap(lambda x, y: (x - y) ** 2, (0, 0), 0)(
                    predicted_Y_dYdX.squeeze(), dslice(Y_dYdX, end_idx, batch_size)
                ),
                [dY],
                axis=1,
            ),
        )
    )
    # L^2 norms of Y and dYdX:
    # 1) Square all entries
    # 2) split into two arrays, Ys and dYdXs
    # 3) sum the splits [:, 0:dY], [:, dY:], sum axis = 1
    return jnp.array(
        [
            jnp.mean(Yhat_L2_norms),
            jnp.mean(dYhatdX_L2_norms),
            jnp.mean(
                __normalize_values(
                    Yhat_L2_norms, dslice(Y_L2_norms, end_idx, batch_size)
                )
            ),
            jnp.mean(
                __normalize_values(
                    dYhatdX_L2_norms, dslice(dYdX_L2_norms, end_idx, batch_size)
                )
            ),
        ]
    )


@eqx.filter_jit
def batch_mean_h1_seminorm_l2_errors_and_norms(
    nn, dM, batch_size, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms, end_idx
):
    # Side effect: jitting a function
    predicted_Y, predicted_dYdX = __value_and_jacrev(nn, dslice(X, end_idx, batch_size))
    Yhat_L2_norms, dYhatdX_L2_norms = vmap(jax.jit(lambda x: jnp.linalg.norm(x) ** 2))(
        predicted_Y.squeeze() - dslice(Y, end_idx, batch_size)
    ), vmap(jax.jit(lambda x: jnp.linalg.norm(x) ** 2))(
        predicted_dYdX.squeeze() - dslice(dYdX, end_idx, batch_size)
    )
    # vmap(, axis=(0,0))(predicted_dYdX.squeeze(), dslice(dYdX, end_idx, batch_size))
    # jnp.sum(
    #     optax.l2_loss(predicted_Y.squeeze(), dslice(Y, end_idx, batch_size)),
    #     axis=1,
    # )*2, jnp.sum(
    #     optax.l2_loss(
    #         predicted_dYdX.squeeze(),
    #         dslice(dYdX, end_idx, batch_size),
    #     ),
    #     # * dM,
    #     axis=(1, 2),
    # )*2
    return jnp.array(
        [
            jnp.mean(Yhat_L2_norms),
            jnp.mean(dYhatdX_L2_norms),
            jnp.mean(
                __normalize_values(
                    Yhat_L2_norms, dslice(Y_L2_norms, end_idx, batch_size)
                )
            ),
            jnp.mean(
                __normalize_values(
                    dYhatdX_L2_norms, dslice(dYdX_L2_norms, end_idx, batch_size)
                )
            ),
        ]
    )


@eqx.filter_jit
def batched_mean_l2_errors_and_norms(batch_size, nn, X, Y, Y_L2_norms, end_idx):
    one_over_n_batches = 1.0 / batch_size

    # No side effects

    predicted_Y = vmap(nn)(dslice(X, end_idx, batch_size))
    mse_i = jnp.mean(
        optax.l2_loss(predicted_Y.squeeze(), dslice(Y, end_idx, batch_size)),
        axis=1,
    )

    return (
        end_idx + batch_size,
        one_over_n_batches * jnp.mean(mse_i),
        one_over_n_batches
        * jnp.mean(__normalize_values(mse_i, dslice(Y_L2_norms, end_idx, batch_size))),
    )


def create_mean_l2_errors_and_norms(batch_size):
    # No side effects

    one_over_n_batches = 1.0 / batch_size

    @eqx.filter_jit
    def __mean_l2_errors_and_norms(nn, X, Y, Y_L2_norms, end_idx):
        # No side effects

        predicted_Y = vmap(nn)(dslice(X, end_idx, batch_size))
        mse_i = jnp.mean(
            optax.l2_loss(predicted_Y.squeeze(), dslice(Y, end_idx, batch_size)),
            axis=1,
        )

        #     acc_l2 = 1.0 - jnp.sqrt(rel_mse)
        # acc_h1 = 1.0 - jnp.sqrt(rel_msje)
        # mean_h1_seminorm_loss = mse + msje
        return (
            end_idx + batch_size,
            one_over_n_batches * jnp.mean(mse_i),
            one_over_n_batches
            * jnp.mean(
                __normalize_values(mse_i, dslice(Y_L2_norms, end_idx, batch_size))
            ),
        )

    return __mean_l2_errors_and_norms


@eqx.filter_jit
def __mean_h1_seminorm_loss(
    dM: int, nn: eqx.nn, input_X: jax.Array, actual_Y: jax.Array, actual_dYdX: jax.Array
):
    predicted_Y, predicted_dYdX = __value_and_jacrev(nn, input_X)
    return (
        jnp.mean(optax.l2_loss(predicted_Y.squeeze(), actual_Y))
        + jnp.mean(optax.l2_loss(predicted_dYdX.squeeze(), actual_dYdX)) * dM * 10
    )


# def squared_error(a, b):


@eqx.filter_jit
def __mean_h1_seminorm_loss_flattened(
    nn: eqx.nn,
    input_X: jax.Array,
    actual_Y_dYdX: jax.Array,
):
    predicted_Y_dYdX = __value_and_jacrev_flattened(nn, input_X)
    return jnp.mean(optax.l2_loss(predicted_Y_dYdX.squeeze(), actual_Y_dYdX))
    # vmap(squared_error, axis = (0,0))(predicted_Y_dYdX.squeeze(), actual_Y_dYdX)


def create_mean_h1_seminorm_loss(dM: int) -> Callable:
    @eqx.filter_jit
    def mean_h1_seminorm_loss(
        nn: eqx.nn, input_X: jax.Array, actual_Y: jax.Array, actual_dYdX: jax.Array
    ):
        predicted_Y, predicted_dYdX = __value_and_jacrev(nn, input_X)
        return (
            jnp.mean(optax.l2_loss(predicted_Y.squeeze(), actual_Y))
            + jnp.mean(optax.l2_loss(predicted_dYdX.squeeze(), actual_dYdX)) * dM * 5
        )

    return mean_h1_seminorm_loss


def create_mean_h1_seminorm_loss_more_correct(dM: int) -> Callable:
    @eqx.filter_jit
    def mean_h1_seminorm_loss(
        nn: eqx.nn, input_X: jax.Array, actual_Y: jax.Array, actual_dYdX: jax.Array
    ):
        predicted_Y, predicted_dYdX = __value_and_jacrev(nn, input_X)
        return (
            jnp.mean(jnp.sum(optax.l2_loss(predicted_Y.squeeze(), actual_Y), axis=1))
            + jnp.mean(
                jnp.sum(
                    optax.l2_loss(predicted_dYdX.squeeze(), actual_dYdX), axis=(1, 2)
                )
            )
            * 5
        )

    return mean_h1_seminorm_loss


@eqx.filter_jit
def __mean_l2_norm_loss(nn: eqx.nn, input_X: jax.Array, actual_Y: jax.Array):
    # No side effects

    predicted_Y = nn(input_X)
    return jnp.mean(optax.l2_loss(predicted_Y.squeeze(), actual_Y))


__grad_mean_h1_seminorm_loss = eqx.filter_grad(__mean_h1_seminorm_loss)
__grad_mean_l2_norm_loss = eqx.filter_grad(__mean_l2_norm_loss)

create_grad_mean_h1_seminorm_loss = lambda dM: eqx.filter_grad(
    create_mean_h1_seminorm_loss(dM)
)

__grad_mean_h1_seminorm_loss_flattened = eqx.filter_grad(__mean_l2_norm_loss)

grad_mean_h1_seminorm_loss_flattened = eqx.filter_grad(
    __mean_h1_seminorm_loss_flattened
)


@eqx.filter_jit
def take_l2_step(
    optimizer_updater, optimizer_state, nn: eqx.Module, X: jax.Array, Y: jax.Array
):
    updates, optimizer_state = optimizer_updater(
        __grad_mean_l2_norm_loss(dM, nn, X, Y), optimizer_state
    )
    return optimizer_state, eqx.apply_updates(nn, updates)


@eqx.filter_jit
def take_h1_step(
    optimizer_updater,
    optimizer_state,
    dM,
    nn: eqx.Module,
    X: jax.Array,
    Y: jax.Array,
    dYdX: jax.Array,
):
    updates, optimizer_state = optimizer_updater(
        __grad_mean_h1_seminorm_loss(dM, nn, X, Y, dYdX), optimizer_state
    )
    return optimizer_state, eqx.apply_updates(nn, updates)


@jax.jit
def __normalize_values(scores, normalizers):  # store L2NormY, L2NormdYdX
    # No side effects
    return scores / normalizers


import numpy as np


def compute_h1_loss_metrics_flat(
    nn, dY, batch_size, n_batches, X, Y_dYdX, Y_L2_norms, dYdX_L2_norms
) -> Callable:
    # DOCUMENT ME
    end_idx, errors = 0, np.zeros((4,), dtype=np.float64)
    # np.accumulatemean(  ,  end_idx +=batch_size   )
    for _ in range(n_batches):
        errors += batch_mean_h1_seminorm_l2_errors_and_norms_flat(
            nn, dY, batch_size, X, Y_dYdX, Y_L2_norms, dYdX_L2_norms, end_idx
        )
        end_idx += batch_size
    errors /= n_batches
    acc_l2, acc_h1 = 1.0 - np.sqrt(errors[2:])
    return acc_l2, acc_h1, np.sum(errors[0:2])


def compute_h1_loss_metrics(
    nn, dM, batch_size, n_batches, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms
) -> Callable:
    # DOCUMENT ME
    end_idx, errors = 0, np.zeros((4,), dtype=np.float64)
    for _ in range(n_batches):
        errors += batch_mean_h1_seminorm_l2_errors_and_norms(
            nn, dM, batch_size, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms, end_idx
        )
        end_idx += batch_size
    errors /= n_batches
    acc_l2, acc_h1 = 1.0 - np.sqrt(errors[2:])
    return acc_l2, acc_h1, np.sum(errors[0:2])


# @eqx.filter_jit
def batched_compute_h1_loss_metrics(
    nn, dM: int, batch_size: int, n_batches, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms
) -> Callable:
    nn_arr_part, nn_static_part = eqx.partition(nn, eqx.is_inexact_array)
    one_over_n_batches = 1.0 / n_batches

    # Side effect: jitting a function
    def __batched_mean_h1_seminorm_l2_errors_and_norms(carry, end_idx):
        nn_arr_part, partials = carry
        nn = eqx.combine(nn_arr_part, nn_static_part)  # just for understanding
        predicted_Y, predicted_dYdX = __value_and_jacrev(
            nn, dslice(X, end_idx, batch_size)
        )
        mse_i, msje_i = jnp.mean(
            optax.l2_loss(predicted_Y.squeeze(), dslice(Y, end_idx, batch_size)),
            axis=1,
        ), jnp.mean(
            optax.l2_loss(
                predicted_dYdX.squeeze(),
                dslice(dYdX, end_idx, batch_size),
            )
            * dM,
            axis=(1, 2),
        )
        nn_arr_part, _ = eqx.partition(nn, eqx.is_inexact_array)
        partials = partials.at[0].add(jnp.mean(mse_i))
        partials = partials.at[1].add(jnp.mean(msje_i))
        partials = partials.at[2].add(
            jnp.mean(__normalize_values(mse_i, dslice(Y_L2_norms, end_idx, batch_size)))
        )  # log mse - log Y_L2_norms
        partials = partials.at[3].add(
            jnp.mean(
                __normalize_values(msje_i, dslice(dYdX_L2_norms, end_idx, batch_size))
            )
        )

        return (nn_arr_part, partials), [0]

    results = jax.lax.scan(
        __batched_mean_h1_seminorm_l2_errors_and_norms,
        (nn_arr_part, jnp.array([0.0, 0.0, 0.0, 0.0])),
        jnp.arange(0, n_batches, batch_size, dtype=jnp.uint32),
    )[0][1]
    return (
        1.0 - jnp.sqrt(results.at[2].get() * one_over_n_batches),
        1.0 - jnp.sqrt(results.at[3].get() * one_over_n_batches),
        (results.at[0].get() + results.at[1].get()) * one_over_n_batches,
    )


def create_compute_h1_loss_metrics(dM: int, batch_size) -> Callable:
    mean_h1_seminorm_errors_and_norms = create_mean_h1_seminorm_l2_errors_and_norms(
        dM, batch_size
    )

    def compute_loss_metrics(nn, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms, n_batches):
        # DOCUMENT ME
        mse = 0.0
        msje = 0.0
        rel_mse = 0.0
        rel_msje = 0.0
        end_idx = 0

        for _ in range(n_batches):
            end_idx, a, b, c, d = mean_h1_seminorm_errors_and_norms(
                nn, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms, end_idx
            )
            mse += a
            msje += b
            rel_mse += c
            rel_msje += d

        acc_l2 = 1.0 - jnp.sqrt(rel_mse / n_batches)
        acc_h1 = 1.0 - jnp.sqrt(rel_msje / n_batches)
        mean_h1_seminorm_loss = (mse + msje) / n_batches
        return acc_l2, acc_h1, mean_h1_seminorm_loss

    return compute_loss_metrics
