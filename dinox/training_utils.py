import time
from typing import Any, Callable, Dict, Iterable, Literal, Tuple

import jax.numpy as jnp
import jax.tree_util
import optax
from equinox import Module as eqxModule
from equinox import apply_updates as eqx_apply_updates
from equinox import filter as eqx_filter
from equinox import filter_jit
from equinox import is_inexact_array as eqx_is_inexact_array
from jax import Array as jax_Array
from jax import device_put as jax_device_put
from jax.tree_util import PyTreeDef

from .data_loading import (
    check_batch_size_validity,
    load_encoded_training_validation_and_testing_data,
)
from .equinox_nn_factories import EquinoxMLPWrapper
from .losses import (
    cpu_compute_bochner_relative_errors,
    compute_bochner_relative_errors,
    vectorized_grad_H1_Bochner_loss,
    vectorized_grad_L2_Bochner_loss,
)


def create_permuter(N: int, cp_random_seed: int = None) -> Callable:
    """
    Creates a callable to permute JAX arrays using permutation indices from CuPy.

    Parameters:
        N (int): Size for generating permutation indices.
        cp_random_seed (int, optional): Seed for reproducible permutation.

    Returns:
        Callable: A function that, given JAX arrays, returns their permuted versions.

    Example:
        >>> permute = create_array_permuter_flat(100, cp_random_seed=42)
        >>> X_permuted, Y_permuted = permute(X, Y)
    """
    import cupy as cp
    from jax.dlpack import from_dlpack as dlpack2jax
    from jax.dlpack import to_dlpack as jax2dlpack

    indices = cp.arange(N)
    if cp_random_seed is not None:
        cp.random.seed(cp_random_seed)

    def permute_arrays(*arrays: Iterable[jax_Array]) -> Tuple[jax_Array, ...]:
        perm = cp.random.permutation(indices)
        # Use DLPack to transfer data between CuPy and JAX and apply permutation
        return (dlpack2jax(cp.from_dlpack(jax2dlpack(arr))[perm]) for arr in arrays)

    return permute_arrays


def create_slicer(*, batch_size: int, num_input_outputs: int) -> Callable:
    """
    Returns a slicing function that extracts a batch from each array using a fixed batch_size and start index.
    """
    from jax.lax import dynamic_slice_in_dim as jittable_slice

    if num_input_outputs == 2:

        @filter_jit
        def slice_data(X: jax_Array, out_1: jax_Array, end_idx: int) -> Tuple[int, jax_Array, jax_Array, jax_Array]:
            """
            Slices X and out_1 from end_idx for batch_size items.

            Returns:
                Updated end index and the slices of X and out_1.
            """
            return (
                end_idx + batch_size,
                jittable_slice(X, end_idx, batch_size),
                jittable_slice(out_1, end_idx, batch_size),
            )

    elif num_input_outputs == 3:

        @filter_jit
        def slice_data(
            X: jax_Array, out_1: jax_Array, out_2: jax_Array, end_idx: int
        ) -> Tuple[int, jax_Array, jax_Array, jax_Array]:
            """
            Slices X, out_1, and out_2 from end_idx for batch_size items.

            Returns:
            Updated end index and the slices of X, out_1, and out_2.
            """
            return (
                end_idx + batch_size,
                jittable_slice(X, end_idx, batch_size),
                jittable_slice(out_1, end_idx, batch_size),
                jittable_slice(out_2, end_idx, batch_size),
            )

    return slice_data


def create_optax_optimization_stepper(
    *,
    LOSS_NAME: Literal["L2", "H1"],  # "score", "hybrid", "h1hybrid", "meanVar", "meanEnt", "fisher",
    nn: eqxModule,
    OPTAX_OPTIMIZER_NAME: str,
    learning_rate: float | optax.Schedule,
) -> Tuple[Callable, optax.OptState, PyTreeDef, eqxModule]:
    """
    Creates a jitted optimization stepper for a neural network using an Optax optimizer.

    Parameters:
        LOSS_NAME (Literal["L2", "H1"]): Chooses the gradient function.
        nn (eqxModule): Neural network to optimize.
        OPTAX_OPTIMIZER_NAME (str): Name of the Optax optimizer.
        learning_rate (float | optax.Schedule): Learning rate.

    Returns:
        Tuple containing:
            - A jitted step function for updating the network and optimizer state.
            - The initial flattened optimizer state.
            - The flattened neural network.
            - The network's tree definition.
    """
    gradient = {
        "L2": vectorized_grad_L2_Bochner_loss,
        "H1": vectorized_grad_H1_Bochner_loss,
    }.get(LOSS_NAME)
    if gradient is None:
        raise Exception(f"LOSS_NAME={LOSS_NAME} is not currently implemented")

    optimizer = optax.__getattribute__(OPTAX_OPTIMIZER_NAME)(learning_rate=learning_rate)
    optimizer_state = optimizer.init(eqx_filter(nn, eqx_is_inexact_array))

    flat_nn, treedef_nn = jax.tree_util.tree_flatten(nn)
    flat_optimizer_state, treedef_optimizer_state = jax.tree_util.tree_flatten(optimizer_state)
    if LOSS_NAME == "L2":

        @filter_jit
        def take_step(
            flat_optimizer_state: optax.OptState,
            flat_nn: eqxModule,
            X: jax_Array,
            Y: jax_Array,
        ) -> Tuple[optax.OptState, eqxModule]:
            nn = jax.tree_util.tree_unflatten(treedef_nn, flat_nn)
            updates, new_optimizer_state = optimizer.update(
                gradient(nn, X, Y),
                jax.tree_util.tree_unflatten(treedef_optimizer_state, flat_optimizer_state),
                nn,
            )
            new_flat_nn = jax.tree_util.tree_leaves(eqx_apply_updates(nn, updates))
            new_flat_optimizer_state = jax.tree_util.tree_leaves(new_optimizer_state)
            return new_flat_optimizer_state, new_flat_nn

    elif LOSS_NAME == "H1":

        @filter_jit
        def take_step(
            flat_optimizer_state: optax.OptState,
            flat_nn: eqxModule,
            X: jax_Array,
            Y: jax_Array,
            dYdX: jax_Array,
        ) -> Tuple[optax.OptState, eqxModule]:
            nn = jax.tree_util.tree_unflatten(treedef_nn, flat_nn)
            updates, new_optimizer_state = optimizer.update(
                gradient(nn, X, Y, dYdX),
                jax.tree_util.tree_unflatten(treedef_optimizer_state, flat_optimizer_state),
                nn,
            )
            new_flat_nn = jax.tree_util.tree_leaves(eqx_apply_updates(nn, updates))
            new_flat_optimizer_state = jax.tree_util.tree_leaves(new_optimizer_state)
            return new_flat_optimizer_state, new_flat_nn

    return (take_step, flat_optimizer_state, flat_nn, treedef_nn)


def n_epochs(
    *,
    stepper: Callable[..., Tuple[optax.OptState, eqxModule]],
    slicer: Callable,
    nn: eqxModule,
    optimizer_state: optax.OptState,
    random_permuter: Callable[..., Tuple[jnp.ndarray, ...]],
    batches_arange: jnp.ndarray,
    epochs_arange: jnp.ndarray,
    training_data: Tuple[jnp.ndarray, ...],
) -> Tuple[optax.OptState, eqxModule]:
    """
    Executes a training loop over epochs using batched and permuted training data.

    Parameters:
        stepper: Function that updates the optimizer state and neural network using a batch.
        slicer: Function that slices data arrays into batches.
        nn: The neural network model.
        optimizer_state (optax.OptState): Current state of the optimizer.
        random_permuter: Function that randomly permutes the training data arrays.
        batches_arange: Array of epoch indices for iterating over the data.
        batches_arange: Array of batch indices for iterating over the data.
        training_data: Tuple of training arrays (e.g., features, labels).

    Returns:
        Tuple[optax.OptState, eqxModule]: The final optimizer state and the trained model.
    """
    for _ in epochs_arange:
        X, *other_data = random_permuter(*training_data)
        end_idx = 0
        for j in batches_arange:
            end_idx, X_batch, *other_data = slicer(X, *other_data, end_idx)
            optimizer_state, nn = stepper(optimizer_state, nn, X_batch, *other_data)
    return optimizer_state, nn


def load_encoded_data_train_and_test_dino(
    eqx_wrapper: EquinoxMLPWrapper,
    N_MAX_EPOCHS: int,
    STEP_SIZE: float,
    BATCH_SIZE: int,
    OPTAX_OPTIMIZER_NAME: str,
    LOSS_NAME: str,
    N_TRAIN: int,
    REDUCED_DIMS: Tuple[int, int],
    RANDOM_PERMUTATIONS_SEED: int,
    N_VAL: int = 2500,
) -> Tuple[EquinoxMLPWrapper, Dict[str, Any]]:
    print("Setting up training problem...")
    data_keys = {  # harded coded
        "L2": ("encoded_inputs", "encoded_output"),
        "H1": ("encoded_inputs", "encoded_output", "encoded_Jacobians"),
    }.get(LOSS_NAME)
    if data_keys is None:
        raise Exception("Not currently implemented")
    else:
        print("Chosen loss:", LOSS_NAME)
    renaming_dict = {
        "encoded_inputs": "X",
        "encoded_output": "fX",
        "encoded_Jacobians": "dfXdX",
    }
    train_val_test = load_encoded_training_validation_and_testing_data(
        REDUCED_DIMS=REDUCED_DIMS,
        data_keys=data_keys,
        renaming_dict=renaming_dict,
        N_TRAIN=N_TRAIN,
        N_VAL=N_VAL,
    )
    nn, results = train_and_test_dino(
        nn=eqx_wrapper.nn,
        N_MAX_EPOCHS=N_MAX_EPOCHS,
        STEP_SIZE=STEP_SIZE,
        BATCH_SIZE=BATCH_SIZE,
        train_val_test=train_val_test,
        OPTAX_OPTIMIZER_NAME=OPTAX_OPTIMIZER_NAME,
        RANDOM_PERMUTATIONS_SEED=RANDOM_PERMUTATIONS_SEED,
        LOSS_NAME=LOSS_NAME,
    )
    eqx_wrapper.params = nn  # is this right?
    return eqx_wrapper, results


def train_and_test_dino(
    nn: eqxModule,
    N_MAX_EPOCHS: int,
    STEP_SIZE: float,
    BATCH_SIZE: int,
    train_val_test: dict,
    N_EPOCHS_BETWEEN_TEST: int = 200,
    OPTAX_OPTIMIZER_NAME: str = "adam",
    RANDOM_PERMUTATIONS_SEED: int = 0,  # val and test have to have Jacobians in them!!!
    LOSS_NAME: str = "H1",
) -> Tuple[eqxModule, Dict[str, Any]]:
    # Place on GPU!

    if not all(list(arr.devices())[0].platform == "gpu" for arr in train_val_test["train"].values()):
        print("Training data did not yet reside on GPU. Placing on GPU.")
        training_data = tuple(jax_device_put(train_val_test["train"][k]) for k in ("X", "fX", "dfXdX"))
    else:
        training_data = tuple(train_val_test["train"][k] for k in ("X", "fX", "dfXdX"))

    cpu_test_data = train_val_test["test"]
    val_data = train_val_test.get("val")
    if val_data is not None:
        assert all(
            list(arr.devices())[0].platform != "gpu" for arr in val_data.values()
        ), "All arrays in train_val_test['val'] should be on the CPU!"
    assert all(
        list(arr.devices())[0].platform != "gpu" for arr in cpu_test_data.values()
    ), "All arrays in train_val_test['test'] should be on the CPU!"

    # Prepare for training
    N_TRAIN = training_data[0].shape[0]
    n_train_batches = check_batch_size_validity(data_iterable=training_data, batch_size=BATCH_SIZE)
    batches_arange = jnp.arange(n_train_batches)
    N_MAX_OUTER_ITERS = N_MAX_EPOCHS // N_EPOCHS_BETWEEN_TEST

    epochs_arange = jnp.arange(N_EPOCHS_BETWEEN_TEST)
    num_train_steps = N_MAX_EPOCHS * n_train_batches
    lr_schedule = optax.piecewise_constant_schedule(
        init_value=STEP_SIZE,
        boundaries_and_scales={
            int(num_train_steps * 0.75): 0.3,
        },
    )

    slicer = create_slicer(batch_size=BATCH_SIZE, num_input_outputs=len(training_data))
    random_permuter = create_permuter(N_TRAIN, RANDOM_PERMUTATIONS_SEED)
    stepper, optimizer_state, nn, nn_treedef = create_optax_optimization_stepper(
        LOSS_NAME=LOSS_NAME,
        nn=nn,
        OPTAX_OPTIMIZER_NAME=OPTAX_OPTIMIZER_NAME,
        learning_rate=lr_schedule,
    )

    print("Started training...")

    total_epoch_time = 0.0
    total_validation_time = 0.0
    validations = dict()
    results = dict()
    for outer_iter in range(N_MAX_OUTER_ITERS):
        start = time.time()
        optimizer_state, nn = n_epochs(
            slicer=slicer,
            random_permuter=random_permuter,
            training_data=training_data,
            stepper=stepper,
            optimizer_state=optimizer_state,
            nn=nn,
            batches_arange=batches_arange,
            epochs_arange=epochs_arange,
        )
        total_epoch_time += time.time() - start

        # evaluate validation error
        if val_data:
            print("computing validation errors")
            start = time.time()
            validation_errors = compute_bochner_relative_errors(
                jax.tree_util.tree_unflatten(nn_treedef, nn),
                *(val_data[key] for key in ("X", "fX", "dfXdX", "fX_norms", "dfXdX_norms")),
            )
            total_validation_time += time.time() - start
            print(validation_errors)
            # break if validation_loss is not decreasing
    validations["total_validation_time"] = total_validation_time
    # Testing error
    nn_pytree = jax.tree_util.tree_unflatten(nn_treedef, nn)  # is it on GPU or CPU right now?

    results["test_errors"] = cpu_compute_bochner_relative_errors(
        nn_pytree,
        *(cpu_test_data[key] for key in ("X", "fX", "dfXdX", "fX_norms", "dfXdX_norms")),
    )
    results["total_training_time"] = total_epoch_time
    results["total_training_time_minus_validation"] = total_epoch_time - total_validation_time
    return nn_pytree, results
