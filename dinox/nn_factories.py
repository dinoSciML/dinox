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
from typing import Any, Callable, Dict, Tuple

import equinox as eqx
import jax
import jax.nn
import jax.numpy as jnp
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
    activation="gelu"
) -> eqx.Module:
    "DOCUMENT ME"
    return eqx.nn.MLP(
        in_size=input_size,
        out_size=output_size,
        width_size=layer_width,
        depth=depth,
        activation=jax.nn.__dict__[activation],
        key=key,
    )

def instantiate_uninitialized_nn(
    *, key: jr.PRNGKey, nn_config_dict: Dict[str, Any]
) -> eqx.Module:
    "DOCUMENT ME"
    if nn_config_dict["architecture"] == "generic_dense":
        relevant_params = [
            "layer_width",
            "depth",
            "input_size",
            "output_size",
            "activation",
        ]
        return GenericDenseFactory(
            **{k: nn_config_dict[k] for k in relevant_params}, key=key
        )
    else:
        raise ("not implemented")


# This is essentially Xavier initialization
def __truncated_normal(weight: jax.Array, key: jr.PRNGKey) -> jax.Array:
    out, in_ = weight.shape
    stddev = math.sqrt(1 / in_)
    return stddev * jax.random.truncated_normal(
        key, shape=(out, in_), lower=-2, upper=2
    )

# Initialize the linear layers of a Neural Network with `init_fn` and the jax key
def __init_linear_layer_weights(
    model: eqx.Module, init_fn: Callable, key: jr.PRNGKey
) -> eqx.Module:
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [
        x.weight
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
        if is_linear(x)
    ]
    weights = get_weights(model)
    new_weights = [
        init_fn(weight, subkey)
        for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
    ]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model


def instantiate_nn(
    *, key: jr.PRNGKey, nn_config_dict: Dict
) -> Tuple[eqx.Module, jr.PRNGKey]:
    """
    This function sets up the dino network for training
    """
    ################################################################################
    # Set up the neural network
    ################################################################################
    eqx_nn_approximator = instantiate_uninitialized_nn(
        key=key, nn_config_dict=nn_config_dict
    )

    ################################################################################
    # Random seed                                                                  #
    ################################################################################
    permute_key, model_key = jr.split(key, 2)

    ################################################################################
    # Load equinox NN parameter checkpoint (as an initial guess for optimization)  #
    # into equinox NN model (pytrees)                                              #
    ################################################################################
    jax_serialized_params_path = nn_config_dict.get("initial_guess_path")
    if jax_serialized_params_path:
        assert os.path.isfile(
            jax_serialized_params_path
        ), "Trained weights may not exist as specified: " + str(
            jax_serialized_params_path
        )
        eqx_nn_approximator = eqx.tree_deserialise_leaves(
            jax_serialized_params_path, eqx_nn_approximator
        )
    else:
        ################################################################################
        # Random initializaiton of equinox NN layers                                   #
        ################################################################################
        eqx_nn_approximator = __init_linear_layer_weights(
            eqx_nn_approximator, __truncated_normal, model_key
        )
    return eqx_nn_approximator, permute_key
