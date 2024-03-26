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

import sys

# import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax  # optax, eqx are dependencies of dinox

from .data_utilities import (
    create_array_permuter,
    load_data_disk_direct_to_gpu,
    save_to_pickle,
    slice_data,
    split_training_testing_data,
    sub_dict,
)
from .nn_factories import instantiate_nn
from .embed_data import embed_data_in_encoder_decoder_subspaces
from .metrics import (
    compute_l2_loss_metrics,
    create_compute_h1_loss_metrics,
    create_grad_mean_h1_seminorm_loss,
    grad_mean_l2_norm_loss,
)

# TODO: f = create_encoder_decoder_nn_from_embedded_dino(nn_approximator, basis, cobasis)
# Y = f(X)


def train_nn_approximator(
    *,
    untrained_approximator,
    training_data,
    testing_data,
    permute_key,
    training_config_dict,
    training_results_dict,
):
    # DOCUMENT ME, remainder not allowed
    # start_time = time.time()
    ####################################################################################
    # Create variable aliases for readability
    ####################################################################################
    nn = untrained_approximator
    n_epochs = training_config_dict["n_epochs"]
    batch_size = training_config_dict["batch_size"]
    loss_norm_weights = training_config_dict["loss_weights"]
    # shuffle_every_epoch = training_config_dict['shuffle_every_epoch']

    ####################################################################################
    # Check if batch_size evenly divides the testing and training data
    ####################################################################################
    n_train, dM = training_data[0].shape
    n_test = testing_data[0].shape[0]
    n_train_batches, remainder = divmod(n_train, batch_size)
    n_test_batches, remainder_test = divmod(n_test, batch_size)
    if remainder != 0:
        raise (
            f"Warning, the number of training data ({n_train}) does not evenly devide into n_batches={n_train_batches}. Adjust `batch_size` to evenly divide {n_train}."
        )
    if remainder_test != 0:
        raise (
            f"Warning, the number of testing data ({n_test}) does not evenly devide into n_batches={n_test_batches}. Adjust `batch_size` to evenly divide {n_test}."
        )
    permute_testing_training_arrays = create_array_permuter(n_train + n_test, 777)
    permute_training_arrays = create_array_permuter(n_train)
    permute_testing_arrays = create_array_permuter(n_test)
    ####################################################################################
    # Setup the optimization step (H1 Mean Error or MSE)
    ####################################################################################
    if loss_norm_weights[1] == 0.0:
        grad_loss = grad_mean_l2_norm_loss
        compute_loss_metrics = compute_l2_loss_metrics
    else:
        # FUTURE TODO: #do this outside of here so that it doesnt rejit for each problem with the same dM and batchsize? if we have this in an outside loop
        grad_loss = create_grad_mean_h1_seminorm_loss(dM)
        compute_loss_metrics = create_compute_h1_loss_metrics(dM, batch_size)

    @eqx.filter_jit
    def take_step(
        optimizer_state, nn: eqx.Module, X: jax.Array, Y: jax.Array, dYdX: jax.Array
    ):

        updates, optimizer_state = optimizer.update(
            grad_loss(nn, X, Y, dYdX), optimizer_state
        )
        return optimizer_state, eqx.apply_updates(nn, updates)

    ####################################################################################
    # Setup, instantiate and initialize the optax optimizer 						   #
    ####################################################################################
    if True:
        # LR scheduling does not seem to help in this case, but this is how you do it.
        num_train_steps = n_epochs * n_train_batches
        print("Using piecewise lr schedule")
        lr_schedule = optax.piecewise_constant_schedule(
            init_value=training_config_dict["step_size"],
            boundaries_and_scales={
                int(num_train_steps * 0.9): 0.1,
            },
        )
    else:
        lr_schedule = training_config_dict["step_size"]

    optimizer_name = training_config_dict.get("optax_optimizer", "adam")
    optimizer = optax.__getattribute__(optimizer_name)(learning_rate=lr_schedule)
    # Initialize optimizer with eqx state (its pytree of weights)
    optimizer_state = optimizer.init(eqx.filter(nn, eqx.is_inexact_array))

    ####################################################################################
    # Setup data structures for storing training results
    ####################################################################################
    # 			'epoch_time': [],

    ####################################################################################
    # Train the neural network
    ###################################################################################
    metrics_history_train = jnp.empty((n_epochs, 3), dtype=jnp.float32)
    metrics_history_test = jnp.empty((n_epochs, 3), dtype=jnp.float32)

    for epoch in jnp.arange(1, n_epochs + 1):
        if (epoch % 20) == 0:
            print("concatenating")
            test_train_data = [
                jax.lax.concatenate((test, train), 0)
                for test, train in zip(training_data, testing_data)
            ]  # only do this is we can't store double the data in memory (otherwise we will just pass in test_train in the beginning also)
            print("permuting and splitting")
            training_data, testing_data = split_training_testing_data(
                permute_testing_training_arrays(*test_train_data),
                training_config_dict["data"],
            )
        # permute with cupy vs permute_key,
        permuted_training_data = permute_training_arrays(*training_data)
        X, Y, dYdX, _, _ = permuted_training_data
        # start_time = time.time()
        end_idx = 0
        for _ in range(n_train_batches):
            end_idx, X_batch, Y_batch, dYdX_batch = slice_data(
                X, Y, dYdX, batch_size, end_idx
            )
            optimizer_state, nn = take_step(
                optimizer_state, nn, X_batch, Y_batch, dYdX_batch
            )  # loss,
            # could combine Y and dYdX (flattened) into Y_dYdX_data and take the L2 norm that way (but then you would need to scale up one of them by the dimension), this would probably speed things up
        # epoch_time = time.time() - start_time

        # print("The last epoch took", epoch_time, "s")

        # start = time.time()
        # Post-process and compute metrics after each epoch
        metrics_history_train = metrics_history_train.at[epoch - 1].set(
            compute_loss_metrics(nn, *permuted_training_data, n_train_batches)  # N x 3
        )

        permuted_test_data = permute_testing_arrays(*testing_data)

        metrics_history_test = metrics_history_test.at[epoch - 1].set(
            compute_loss_metrics(nn, *permuted_test_data, n_test_batches)  # N x 3
        )
        # metric_time = time.time() - start

        # metrics_history['epoch_time'][epoch] = epoch_time

        print(
            f"train epoch: {epoch}, "
            f"loss: {(metrics_history_train[epoch-1, 2]):.4f}, "
            f"accuracy l2 : {(metrics_history_train[epoch-1, 0] * 100):.4f}, "
            f"accuracy h1 : {(metrics_history_train[epoch-1, 1] * 100):.4f}"
        )
        print(
            f" test epoch: {epoch}, "
            f"loss: {(metrics_history_test[epoch-1, 2]):.4f}, "
            f"accuracy l2: {(metrics_history_test[epoch-1, 0] * 100):.4f}, "
            f"accuracy h1: {(metrics_history_test[epoch-1, 1] * 100):.4f}"
        )
        # print('Max test accuracy L2 = ', 100*np.max(np.array(metrics_history['test_accuracy_l2'])))
        # print('Max test accuracy H1 (semi-norm) = ', 100*np.max(np.array(metrics_history['test_accuracy_h1'])))
        # print('The metrics took', metric_time, 's')

    # metrics_history_train and metrics_history_test are stored as N_iters x 3
    (
        training_results_dict["train_accuracy_l2"],
        training_results_dict["train_accuracy_h1"],
        training_results_dict["train_loss"],
    ) = metrics_history_train.T
    (
        training_results_dict["test_accuracy_h1"],
        training_results_dict["test_accuracy_l2"],
        training_results_dict["test_loss"],
    ) = metrics_history_test.T
    # print("Total time", time.time() - start_time)
    return training_results_dict, nn


def train_dino_in_embedding_space(random_seed, embedded_training_config_dict):
    #################################################################################
    # Create variable aliases for readability										#
    #################################################################################
    config_dict = embedded_training_config_dict

    #################################################################################
    # Load training/testing data from disk, directly onto GPU as jax arrays  		#
    #################################################################################
    config_dict["data"]["N"] = 5000  # TEMPORARY HACK until i know what to do here
    data_config_dict = config_dict["data"]
    data = load_data_disk_direct_to_gpu(data_config_dict)  # Involves Disk I/O

    #################################################################################
    # Embed training data in subspace using encoder/decoder bases/cobases           #
    #################################################################################
    encodec_dict = config_dict["encoder_decoder"]

    # If `data` is not already `reduced` and we want to encode/decode
    # (i.e. use the active subspace)
    if (encodec_dict["encode"] or encodec_dict["decode"]) and data_config_dict.get(
        "reduced_data_filenames"
    ) is None:
        # Involves Disk I/O
        data = embed_data_in_encoder_decoder_subspaces(data, encodec_dict)
        # config_dict['encoder_decoder']['last_layer_bias'] = np.mean(training_data['q_data'],axis = 0)

    #################################################################################
    # Split the data into training/testing data 	                                #
    #################################################################################
    training_data, testing_data = split_training_testing_data(
        data, config_dict["data"], calculate_norms=True
    )

    #################################################################################
    # Set up the neural network and train it										#
    #################################################################################
    training_results_dict = {}
    nn_config_dict = config_dict["nn"]
    nn_config_dict["input_size"] = training_data[0].shape[1]
    nn_config_dict["output_size"] = training_data[1].shape[1]
    untrained_approximator, permute_key = instantiate_nn(
        key=jr.key(random_seed), nn_config_dict=nn_config_dict
    )
    config_dict["training"]["data"] = config_dict[
        "data"
    ]  # hack for test/train splitting
    trained_approximator = train_nn_approximator(
        untrained_approximator=untrained_approximator,
        training_data=training_data,
        testing_data=testing_data,
        permute_key=permute_key,
        training_config_dict=config_dict["training"],
        training_results_dict=training_results_dict,
    )

    #################################################################################
    # Save training metrics results to disk							                #
    #################################################################################
    network_serialization_config_dict = config_dict["network_serialization"]
    save_name = network_serialization_config_dict["network_name"]
    # logger = {'reduced':training_logger} #,'full': final_logger}
    logging_dir = "logging"
    save_to_pickle(Path(logging_dir, save_name), training_results_dict)

    #################################################################################
    # Save neural network parameters to disk (serialize the equinox pytrees)        #
    #################################################################################
    # Involves Disk I/O
    if network_serialization_config_dict["save_weights"]:
        eqx.tree_serialise_leaves(
            f"{network_serialization_config_dict['weights_dir']}{save_name}.eqx",
            trained_approximator,
        )

    #################################################################################
    # Save config file for reproducibility                                          #
    #################################################################################
    cli_dir = "cli"
    # Involves Disk I/O
    save_to_pickle(Path(cli_dir, save_name), config_dict)
