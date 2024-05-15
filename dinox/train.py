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

import time
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from ._data_utilities import (
    create_arrays_permuter,
    load_data_disk_direct_to_gpu,
    load_data_disk_direct_to_gpu_no_jacobians,
    makedirs,
    save_to_pickle,
    slice_flat_data,
    split_training_testing_data_flat,
)
from .embed_data import embed_data_in_encoder_decoder_subspaces
from .losses import (
    mean_flattened_h1_losses,
    mean_l2_losses,
    grad_mean_flattened_h1_norm_loss,
    grad_mean_l2_norm_loss,
)

from .nn_factories import instantiate_nn

# TODO: f = create_encoder_decoder_nn_from_embedded_dino(nn_approximator, basis, cobasis)

def __check_batch_size_divisibility(data_iterable, *, batch_size):
    for data_tuple in data_iterable:
        n_data = data_tuple[0].shape[0]
        n_batches, remainder = divmod(n_data, batch_size)
        if remainder != 0:
            raise ValueError(
                "Adjust `batch_size` to evenly divide training and testing data."
            )

def define_optimization_stepper(
    *,
    loss: Literal["l2", "h1"],
    nn: eqx.Module,
    optax_optimizer_name: str,
    learning_rate: Union[float, optax.Schedule],
    treedef_dict: Optional[Dict] = None,
) -> Tuple[Callable, Any, eqx.Module]:
    """
    Define an optimization stepper for training a neural network using Optax optimizers.

    Parameters
    ----------
    loss : str
        One of 'l2' or 'h1'.
    nn : eqx.Module
        The neural network model to be optimized.
    optax_optimizer_name : str
        Name of the optimizer to be used from the Optax library.
    learning_rate : Union[float, optax.Schedule]
        A fixed learning rate or a learning rate schedule.
    treedef_dict : Optional[Dict]
        A dictionary that, if provided, will store the tree definitions of the neural
        network and optimizer state for efficient JAX operations.

    Returns
    -------
    Tuple[Callable, Any, eqx.Module]
        A tuple containing the step function, the initial optimizer state, and
        potentially a flattened representation of the neural network model. The
        step function is a jitted function that updates the model and optimizer state.

    Raises
    ------
    AttributeError
        If the specified optimizer name is not found in the Optax module.
    """
    gradient = (
        grad_mean_l2_norm_loss if loss == "l2" else grad_mean_flattened_h1_norm_loss
    )
    optimizer = optax.__getattribute__(optax_optimizer_name)(
        learning_rate=learning_rate
    )
    optimizer_state = optimizer.init(eqx.filter(nn, eqx.is_inexact_array))

    if treedef_dict is not None:
        flat_nn, treedef_nn = jax.tree_util.tree_flatten(nn)
        flat_optimizer_state, treedef_optimizer_state = jax.tree_util.tree_flatten(
            optimizer_state
        )
        treedef_dict["treedef_nn"] = treedef_nn

        @eqx.filter_jit
        def take_tree_flattened_step(
            flat_optimizer_state: Any,
            flat_nn: eqx.Module,
            X: jax.Array,
            Y_dYdX: jax.Array,
        ) -> Tuple[Any, eqx.Module]:
            nn = jax.tree_util.tree_unflatten(treedef_nn, flat_nn)
            updates, new_optimizer_state = optimizer.update(
                gradient(nn, X, Y_dYdX),
                jax.tree_util.tree_unflatten(
                    treedef_optimizer_state, flat_optimizer_state
                ),
                nn,
            )
            new_flat_nn = jax.tree_util.tree_leaves(eqx.apply_updates(nn, updates))
            new_flat_optimizer_state = jax.tree_util.tree_leaves(new_optimizer_state)
            return new_flat_optimizer_state, new_flat_nn

        return take_tree_flattened_step, flat_optimizer_state, flat_nn
    else:
        print("using not flat take step")
        @eqx.filter_jit
        def take_step(
            optimizer_state: Any, nn: eqx.Module, X: jax.Array, Y_dYdX: jax.Array
        ) -> Tuple[Any, eqx.Module]:
            updates, new_optimizer_state = optimizer.update(
                gradient(nn, X, Y_dYdX), optimizer_state, nn
            )
            new_nn = eqx.apply_updates(nn, updates)
            return new_optimizer_state, new_nn

        return take_step, optimizer_state, nn


def training_loop(
    *,
    stepper: Callable[[Any, Any, jnp.ndarray, jnp.ndarray], Tuple[Any, Any]],
    slicer: Callable[
        [jnp.ndarray, jnp.ndarray, int, int], Tuple[int, jnp.ndarray, jnp.ndarray]
    ],
    nn: eqx.Module,
    optimizer_state: Any,
    batch_size: int,
    array_permuter: Callable[
        [jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]
    ],
    n_epochs: int,
    data: Tuple[jnp.ndarray, jnp.ndarray],
    testing_data,
    dY,
    tree_def_nn
) -> Any:
    """
    Executes a generic training loop for a given neural network model using
    batched data. Ensures that the batch size evenly divides the number of data samples.

    Parameters
    ----------
    stepper : Callable
        A function to update the optimizer state and model based on the batch data.
    slicer : Callable
        A function to slice the data into batches.
    optimizer_state : Any
        The current state of the optimizer.
    batch_size : int
        The size of each batch of data.
    array_permuter : Callable
        A function to permute the data arrays before batching.
    n_epochs : int
        The number of epochs to train the model.
    data : Tuple[jnp.ndarray, jnp.ndarray]
        A tuple containing the input features and the labels/desired outputs.

    Returns
    -------
    Any
        The trained neural network model.

    Raises
    ------
    ValueError
        If the batch size does not evenly divide the total number of data samples.
    """
    n_batches, remainder = divmod(len(data[0]), batch_size)
    if remainder != 0:
        error_msg = (
            f"The number of data ({len(data[0])}) does not evenly divide into "
            f"n_batches={n_batches}. Adjust `batch_size` to evenly divide {len(data[0])}."
        )
        raise ValueError(error_msg)
    testing_data = [data[0:2500] for data in testing_data]
    for i in jnp.arange(1, n_epochs + 1):
        X, Y_dYdX = array_permuter(*data)

        end_idx = 0
        for j in range(n_batches):
            end_idx, X_batch, Y_dYdX_batch = slicer(X, Y_dYdX, batch_size, end_idx)
            optimizer_state, nn = stepper(optimizer_state, nn, X_batch, Y_dYdX_batch)

        if (i % 100) == 0:
            nn_unflat = jax.tree_util.tree_unflatten(tree_def_nn, nn)
            test_loss = mean_flattened_h1_losses(nn_unflat, dY, *testing_data)
            # test_loss = mean_flattened_h1_losses(nn_unflat, dY, *training_data_batch)s
            print(i, "test_accuracy_l2", test_loss[0])
            print(i, "test_accuracy_h1", test_loss[1], "\n")
    return nn


def train_nn_approximator(
    *,
    untrained_approximator: Any,
    training_data: Tuple[Any, Any],
    testing_data: Tuple[Any, Any],
    permute_seed: int,
    training_config: Dict[str, Any],
) -> Tuple[Any, Dict[str, float]]:
    """
    Train the provided neural network approximator using optax optimizer
    and return the trained network along with a dictionary of training and testing results.

    Parameters
    ----------
    untrained_approximator : Any
        Initial untrained model to be optimized.
    training_data : Tuple[Any, Any]
        Training data comprising features and labels.
    testing_data : Tuple[Any, Any]
        Testing data comprising features and labels.
    permute_seed : int
        int used for random seed by cupy, used for data permutation.
    training_config : Dict[str, Any]
        Dictionary containing training configurations like number of epochs, batch size,
        loss weights, and optimization parameters.

    Returns
    -------
    Tuple[Any, Dict[str, float]]
        A tuple containing the trained neural network model and a dictionary with training
        and testing metrics such as accuracy and loss.

    Side Effects
    ------------
    - Prints training progress and final metrics to stdout.
    """
    start_time = time.time()
    nn = untrained_approximator
    print("Setting up training problem...")
    n_epochs = training_config["nEpochs"]
    batch_size = training_config["batchSize"]
    loss = training_config["loss"]

    n_train, dM = training_data[0].shape
    n_test = testing_data[0].shape[0]

    # Creating array permuter used for shuffling data after each epoch
    __check_batch_size_divisibility(
        (training_data, testing_data), batch_size=batch_size
    )
    training_arrays_permuter = create_arrays_permuter(n_train, permute_seed)

    # Define optimizer
    n_train_batches, _ = divmod(n_train, batch_size)
    num_train_steps = n_epochs * n_train_batches
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=training_config["stepSize"],
        boundaries_and_scales={
            int(num_train_steps * 0.75): 0.3,
            },

    )

    tree_flattened=True
    if tree_flattened:
        treedef_dict = {}
    else:
        treedef_dict = None #current this case isn't built 

    optax_stepper, optimizer_state, nn = (
        define_optimization_stepper(
            loss=loss,
            nn=nn,
            optax_optimizer_name=training_config.get("optaxOptimizer", "adam"),
            learning_rate=lr_schedule,
            treedef_dict=treedef_dict,
        )
    )
    # for flat data
    
    if loss == 'l2':
        testing_data = (testing_data[0],testing_data[2],testing_data[3],testing_data[4])
        dY = int(training_data[2].shape[1] / (dM + 1))
    else:
        dY = int(training_data[1].shape[1] / (dM + 1))
    print("Started training...")
    nn = training_loop(
        stepper=optax_stepper,
        slicer=slice_flat_data,
        nn=nn,
        optimizer_state=optimizer_state,
        batch_size=batch_size,
        array_permuter=training_arrays_permuter,
        n_epochs=n_epochs,
        data=training_data[0:2],
        testing_data=testing_data,
        dY=dY,
        tree_def_nn=treedef_dict['treedef_nn'] if tree_flattened else None
    )
    if tree_flattened:
        nn = jax.tree_util.tree_unflatten(treedef_dict['treedef_nn'], nn)
    print("Done training")

    # Evaluate NN accuracy
    if loss == 'l2':
        training_data = (training_data[0],training_data[2],training_data[3],training_data[4])
    train_loss = mean_flattened_h1_losses(nn, dY, *training_data)
    test_loss = mean_flattened_h1_losses(nn, dY, *testing_data)

    results = {
        "train_accuracy_l2": train_loss[0],
        "train_accuracy_h1": train_loss[1],
        "train_l2_loss": train_loss[2],
        "train_h1_loss": train_loss[3],
        "test_accuracy_l2": test_loss[0],
        "test_accuracy_h1": test_loss[1],
        "test_l2_loss": test_loss[2],
        "test_h1_loss": test_loss[3],
    }
    print("results:", results)
    print("Total time", time.time() - start_time)
    return nn, results


def save_training_results(*, results, nn, config):
    # Save training results
    save_to_pickle(config['training_metrics_path'], results )

    # Save nn parameters/ class parameters
    makedirs(config['nn_weights_path'].parents[0], exist_ok=True)
    eqx.tree_serialise_leaves(config['nn_weights_path'], nn)
    config["nn_config"]['filename'] = config["nn_weights_path"]
    print(config['nn_class_path'])
    save_to_pickle(config['nn_class_path'], config['nn_config'])


def train_nn_in_embedding_space(
    random_seed: int, embedded_training_config: Dict[str, Any]
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a nn in an embedding space and return it along with training results.

    Parameters:
    random_seed (int): Seed for reproducibility.
    embedded_training_config (dict): Configuration for training.

    Returns:
    Tuple[Any, Dict[str, Any]]: The trained model and results dictionary.

    Side effects:
    - Loads data from disk to GPU.
    - May save results to disk during training.
    """

    # Create variable alias for readability
    config = embedded_training_config

    # Load training/testing data from disk, onto GPU as jax arrays
    data_config = config["data"]
    print("Loading data to GPU, takes a few seconds..")
    # if config["training"]["loss"] == "h1":
    data = load_data_disk_direct_to_gpu(data_config)
    # else:
    #     data = load_data_disk_direct_to_gpu_no_jacobians(data_config)

    # Embed training data using encoder/decoder bases/cobases
    encodec_dict = config["encoder_decoder"]
    if (encodec_dict["encoder"] or encodec_dict["decoder"]):
        print("Embedding data in encoder subspace..")
        data = embed_data_in_encoder_decoder_subspaces(data, encodec_dict)

    print("Splitting data into training/testing data")
    data_config['jacobian'] = (config["training"]["loss"] == "h1")
    training_data, testing_data = split_training_testing_data_flat(
        data, data_config, calculate_norms=True
    )
    # Set up the neural network and train it
    nn_config = config["nn"]
    nn_config["input_size"] = training_data[0].shape[1]
    if config["training"]["loss"] == "h1": 
        # If loss is h1, the data is a concatenation of Y & flattened dYdX
        # the networks output should be the dimension of Y
        nn_config["output_size"] = int(
            training_data[1].shape[1] / (nn_config["input_size"] + 1)
        )
    else:
        nn_config["output_size"] = training_data[1].shape[1]

    print("Instantiating the NN, takes a few seconds")
    untrained_approximator, permute_key = instantiate_nn(
        nn_config=nn_config, key=jr.key(random_seed)
    )
    trained_approximator, results_dict = train_nn_approximator(
        untrained_approximator=untrained_approximator,
        training_data=training_data,
        testing_data=testing_data,
        permute_seed=random_seed,
        training_config=config["training"],
    )

    return trained_approximator, results_dict
