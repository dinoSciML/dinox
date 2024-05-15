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

import math
import os
from typing import Any, Callable, Dict, List, Tuple

import equinox as eqx
import jax
import jax.nn
import jax.random as jr

# This file contains utilities for initializing equinox (build on jax) neural networks
# TODO: implement other Neural Networks
# def TransformerFactory(),
# def ResNetFactory()
# def CNNFactory()


def GenericDenseFactory(
    *,
    layer_width: int,
    depth: int,
    input_size: int,
    output_size: int,
    key: jr.PRNGKey,
    activation: str = "gelu"
) -> eqx.Module:
    """
    Convenience function to create a multi-layer perceptron (MLP) with specified parameters using the Equinox library.

    Parameters
    ----------
    layer_width : int
        The width of each hidden layer (number of neurons per hidden layer).
    depth : int
        The number of hidden layers in the MLP.
    input_size : int
        The size of the input layer (number of input features).
    output_size : int
        The size of the output layer (number of output features).
    key : jr.PRNGKey
        A JAX random key used for initializing the weights of the MLP.
    activation : str, optional
        The activation function to use in each hidden layer, specified as a string that
        must match an activation function name in `jax.nn`. Default is 'gelu'.

    Returns
    -------
    eqx.Module
        An Equinox module with uninitialized weights.

    Notes
    -----
    The function constructs an MLP where all hidden layers have the same width.
    The network uses specified activation function, which is applied after each hidden layer.
    Weight initialization and activation function are determined by the JAX library.

    Examples
    --------
    >>> key = jax.random.PRNGKey(0)
    >>> model = GenericDenseFactory(
            layer_width=128, depth=3, input_size=784, output_size=10, key=key, activation='relu'
        )
    """
    return eqx.nn.MLP(
        in_size=input_size,
        out_size=output_size,
        width_size=layer_width,
        depth=depth,
        activation=jax.jit(jax.nn.__dict__[activation]),
        key=key,
    )


def define_uninitialized_nn(
    *, nn_config: Dict[str, Any], key: jr.PRNGKey = jr.key(0)
) -> eqx.Module:
    """
    Convenience function. Defines an uninitialized neural network based on the provided configuration
    dictionary and a random key. Currently, only supports the initialization of a generic
    dense network architecture. Should add more network architectures here.

    Parameters
    ----------
    nn_config : Dict[str, Any]
        A dictionary containing the neural network configuration, which must include the
        architecture type and relevant parameters for the specific architecture.
    key : jr.PRNGKey, optional
        A JAX random key used for the random initialization of weights, defaulting to a
        key generated with seed 0.

    Returns
    -------
    eqx.Module
        An uninitialized Equinox module of the specified architecture type.

    Raises
    ------
    ValueError
        If the specified architecture is not implemented.

    Notes
    -----
    The function currently supports only 'genericDense' architecture. If `architecture`
    in `nn_config_dict` is 'genericDense', it initializes a Generic Dense neural network
    factory with the parameters specified in `nn_config_dict`. If other architectures are
    specified, it raises a ValueError indicating that the architecture is not implemented.
    """
    if nn_config["architecture"] == "genericDense":
        relevant_params = [
            "layer_width",
            "depth",
            "input_size",
            "output_size",
            "activation",
        ]
        return GenericDenseFactory(
            **{k: nn_config[k] for k in relevant_params}, key=key
        )
    else:
        raise ValueError("Architecture not implemented")


# This is essentially Xavier initialization
def __truncated_normal(weight: jax.Array, key: jr.PRNGKey) -> jax.Array:
    """
    Generates weights initialized according to a truncated normal distribution
    with a specified standard deviation and limits, determined by the input
    weight shape.

    Parameters
    ----------
    weight : jax.Array
        A JAX array whose shape is used to define the shape of the output array
        and to calculate the standard deviation for the truncated normal distribution.
        The shape of `weight` should be (output_dim, input_dim), where `input_dim` is used
        to calculate the standard deviation as `sqrt(1 / input_dim)`.
    key : jr.PRNGKey
        A JAX random key used to generate the truncated normal values.

    Returns
    -------
    jax.Array
        An array of the same shape as `weight`, where each element is drawn from a
        truncated normal distribution with calculated standard deviation and truncated
        between -2 and 2.

    Notes
    -----
    The truncated normal distribution is used typically for initializing weights in neural
    networks to prevent large initial weights. This implementation limits the weights
    to be between -2 and 2 standard deviations of the mean, which is set to 0.
    """
    out, in_ = weight.shape
    stddev = math.sqrt(1 / in_)
    return stddev * jax.random.truncated_normal(
        key, shape=(out, in_), lower=-2, upper=2
    )


def __get_nn_weights(model: eqx.Module) -> List[jax.Array]:
    """
    Retrieves the weights from linear layers of an Equinox neural network model.

    This function traverses the model's computational tree and extracts the weight
    matrices from each linear layer defined in the model. It specifically checks
    for instances of `eqx.nn.Linear` within the model's leaf nodes.

    Parameters
    ----------
    model : eqx.Module
        The Equinox neural network model from which to extract the weights.

    Returns
    -------
    List[jax.Array]
        A list of JAX arrays, each representing the weight matrix of a linear
        layer in the model.

    Examples
    --------
    >>> import equinox as eqx
    >>> model = eqx.nn.MLP(in_size=784, out_size=10, width_size=256, depth=3, activation=jax.nn.relu)
    >>> weights = __get_nn_weights(model)
    >>> print([w.shape for w in weights])
    [(256, 784), (256, 256), (10, 256)]
    """
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    return [
        x.weight
        for x in jax.tree_util.tree_leaves(model, is_leaf=is_linear)
        if is_linear(x)
    ]


# Initialize the linear layers of a Neural Network with `init_fn` and the jax key
def __init_linear_layer_weights(
    model: eqx.Module,
    init_fn: Callable[[eqx.nn.Linear, jr.PRNGKey], eqx.nn.Linear],
    key: jr.PRNGKey,
) -> eqx.Module:
    """
    Initializes the weights of linear layers within a given model using a specified
    initialization function and a JAX random key.

    Parameters
    ----------
    model : eqx.Module
        The model containing linear layers whose weights need initialization.
    init_fn : Callable
        A function that takes a layer and a random key and returns the layer with
        initialized weights. This function should be compatible with the linear layer's weight shape.
    key : jr.PRNGKey
        A JAX random key used for generating subkeys for initializing each linear layer's weights.

    Returns
    -------
    eqx.Module
        The model with initialized weights for all linear layers.

    Notes
    -----
    This function identifies all linear layers within the provided model, then initializes
    their weights using the provided initialization function and splits of the provided key.
    This process ensures that all linear layers' weights are initialized independently
    but based on the same initial random state for reproducibility.
    """
    # Retrieve the current weights of all linear layers
    weights = __get_nn_weights(model)

    # Initialize new weights for each linear layer using the provided init function
    new_weights = [
        init_fn(weight, subkey)
        for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
    ]

    # Update the model with new weights
    new_model = eqx.tree_at(__get_nn_weights, model, new_weights)
    return new_model


def instantiate_nn(
    *, nn_config: Dict, key: jr.PRNGKey = jr.key(0)
) -> Tuple[eqx.Module, jr.PRNGKey]:
    """
    Initializes a DINO neural network from a configuration dictionary and a random key.
    If a checkpoint path is provided and valid, it loads parameters from it; otherwise,
    it initializes the network with random weights.

    Parameters
    ----------
    nn_config : Dict
        A dictionary containing the neural network configuration, including possibly
        a checkpoint path for pretrained weights.
    key : jr.PRNGKey, optional
        A JAX random key used for parameter initialization (default is generated with seed 0).

    Returns
    -------
    Tuple[eqx.Module, jr.PRNGKey]
        A tuple containing the initialized neural network and a new permutation key.

    Raises
    ------
    AssertionError
        If a checkpoint path is provided but the file does not exist.

    Notes
    -----
    The function splits the provided key to create a new permute key and potentially
    another key for random initialization of weights. This process and network instantiation
    depend on the presence and validity of a specified checkpoint path.
    """
    # Set up the neural network
    eqx_nn_approximator = define_uninitialized_nn(
        nn_config=nn_config, key=key #key is unused
    )

    # Random seed
    permute_key, model_key = jr.split(key, 2)

    # Load equinox NN parameter checkpoint (as an initial guess for optimization)
    # into equinox NN model (pytrees)
    jax_serialized_params_path = nn_config.get("filename")
    if jax_serialized_params_path:
        assert os.path.isfile(
            jax_serialized_params_path
        ), "Trained weights may not exist as specified: " + str(
            jax_serialized_params_path
        )
        eqx_nn_approximator = eqx.tree_deserialise_leaves(
            jax_serialized_params_path, eqx_nn_approximator
        )
        print("Successfully loaded equinox NN from disk")
    else:
        # Random initialization of equinox NN layers
        eqx_nn_approximator = __init_linear_layer_weights(
            eqx_nn_approximator, __truncated_normal, model_key
        )

    return eqx_nn_approximator, permute_key

