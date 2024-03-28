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
from functools import partial
from jax import vjp, vmap
from jax.lax import dynamic_slice_in_dim as dslice

__all__ = ["grad_mean_l2_norm_loss",
           "grad_mean_h1_seminorm_loss",
        #    "compute_l2_loss_metrics",
           "create_compute_h1_loss_metrics",
           "batched_compute_h1_loss_metrics"
           ""]



def __value_and_jacrev(f, xs):
    # No side effects
    _, pullback = vjp(f, xs[0])
    basis = jnp.eye(_.size, dtype=_.dtype)

    @jax.jit
    def value_and_jacrev_x(x):
        y, pullback = vjp(f, x)
        jac = vmap(pullback)(basis)
        return y, jac[0]  # There is only one jacobian matrix here, so we extract it

    return vmap(value_and_jacrev_x)(xs)

@eqx.filter_jit
def __batched_mean_h1_seminorm_l2_errors_and_norms(carry, end_idx):
    dM, batch_size, n_batches, nn, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms, partial_mse, partial_msje, partial_rel_mse, partial_rel_msje = carry
    one_over_n_batches = 1.0 / n_batches
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
    return ( nn, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms,
        partial_mse + one_over_n_batches * jnp.mean(mse_i),
        partial_msje + one_over_n_batches * jnp.mean(msje_i),
        partial_rel_mse + one_over_n_batches
        * jnp.mean(
            __normalize_values(mse_i, dslice(Y_L2_norms, end_idx, batch_size))
        ),
        partial_rel_msje + one_over_n_batches
        * jnp.mean(
            __normalize_values(msje_i, dslice(dYdX_L2_norms, end_idx, batch_size))
        )
    )

def __create_mean_h1_seminorm_l2_errors_and_norms(dM, batch_size):
    # Side effect: jitting a function

    one_over_n_batches = 1.0 / batch_size

    @eqx.filter_jit
    def __mean_h1_seminorm_l2_errors_and_norms(
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
            one_over_n_batches * jnp.mean(mse_i),
            one_over_n_batches * jnp.mean(msje_i),
            one_over_n_batches
            * jnp.mean(
                __normalize_values(mse_i, dslice(Y_L2_norms, end_idx, batch_size))
            ),
            one_over_n_batches
            * jnp.mean(
                __normalize_values(msje_i, dslice(dYdX_L2_norms, end_idx, batch_size))
            ),
        )

    return __mean_h1_seminorm_l2_errors_and_norms

@partial(eqx.filter_jit, static_argnames=['batch_size'])
def batched_mean_l2_errors_and_norms(
    batch_size,nn, X, Y, Y_L2_norms, end_idx):
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
        one_over_n_batches * jnp.mean(
            __normalize_values(mse_i, dslice(Y_L2_norms, end_idx, batch_size))
        )
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
            one_over_n_batches * jnp.mean(
                __normalize_values(mse_i, dslice(Y_L2_norms, end_idx, batch_size))
            )
        )
    return __mean_l2_errors_and_norms

@partial(eqx.filter_jit, static_argnames=['dM'])
def __mean_h1_seminorm_loss(
    dM: int,
    nn: eqx.nn, input_X: jax.Array, actual_Y: jax.Array, actual_dYdX: jax.Array
):
    predicted_Y, predicted_dYdX = __value_and_jacrev(nn, input_X)
    return (
        jnp.mean(optax.l2_loss(predicted_Y.squeeze(), actual_Y))
        + jnp.mean(optax.l2_loss(predicted_dYdX.squeeze(), actual_dYdX)) * dM
    )



@eqx.filter_jit
def __mean_l2_norm_loss(nn: eqx.nn, input_X: jax.Array, actual_Y: jax.Array):
    # No side effects

    predicted_Y = nn(input_X)
    return jnp.mean(optax.l2_loss(predicted_Y.squeeze(), actual_Y))


grad_mean_h1_seminorm_loss = eqx.filter_grad(__mean_h1_seminorm_loss)
grad_mean_l2_norm_loss = eqx.filter_grad(__mean_l2_norm_loss)


@jax.jit
def __normalize_values(scores, normalizers):  # store L2NormY, L2NormdYdX
    # No side effects

    return scores / normalizers


def create_compute_h1_loss_metrics(dM: int, batch_size) -> Callable:
    batch_size_mean_h1_seminorm_errors_and_norms = \
        __create_mean_h1_seminorm_l2_errors_and_norms(
        dM, batch_size
    )

    # we can jit this
    def compute_loss_metrics(nn, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms, n_batches):
        # DOCUMENT ME

        # fill an array jax
        mse = 0.0
        msje = 0.0
        rel_mse = 0.0
        rel_msje = 0.0
        # errors = jnp.zeros((4,1))
        end_idx = 0

        # use lax.scan
        for _ in range(n_batches):
            end_idx, a, b, c, d = batch_size_mean_h1_seminorm_errors_and_norms(
                nn, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms, end_idx
            )
            mse += a
            msje += b
            rel_mse += c
            rel_msje += d

        acc_l2 = 1.0 - jnp.sqrt(rel_mse)
        acc_h1 = 1.0 - jnp.sqrt(rel_msje)
        mean_h1_seminorm_loss = mse + msje
        return acc_l2, acc_h1, mean_h1_seminorm_loss

    return compute_loss_metrics


@eqx.filter_jit(donate=['X', 'Y', 'dYdX', 'Y_L2_norms', 'dYdX_L2_norms'])
def batched_compute_h1_loss_metrics(
        dM: int,
        batch_size: int,
        n_batches,
        nn, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms ) -> Callable:
    results = jax.lax.scan(__batched_mean_h1_seminorm_l2_errors_and_norms, ((dM, batch_size, n_batches, nn, X, Y, dYdX, Y_L2_norms, dYdX_L2_norms, 0.0,0.0,0.0,0.0), 0), jnp.arange(0, n_batches, batch_size))[0]
    mse, msje, rel_mse, rel_msje = results[7:11]

    #finally:
    acc_l2 = 1.0 - jnp.sqrt(rel_mse)
    acc_h1 = 1.0 - jnp.sqrt(rel_msje)
    mean_h1_seminorm_loss = mse + msje
    return acc_l2, acc_h1, mean_h1_seminorm_loss
