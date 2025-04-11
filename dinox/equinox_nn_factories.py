import json
import math
from pathlib import Path
from typing import Any, Dict, List

import equinox as eqx
import jax
import jax.random as jr
from pydantic import (BaseModel, ConfigDict, field_serializer, field_validator,
                      model_serializer, model_validator)


# equinox_nn_factories
class EquinoxMLPConfig(BaseModel, validate_assignment=True):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=False)

    input_dimension: int
    output_dimension: int
    random_initializer_key: Any
    activation_name: str
    layer_width: int
    depth: int

    @field_serializer("random_initializer_key")
    def serialize_random_initializer_key(self, key: jax.Array, _info) -> list:
        """Convert JAX PRNGKey to a list."""
        # Note: adjust this conversion if needed
        return jax.random.key_data(key).tolist()[1]

    @field_validator("random_initializer_key", mode="before")
    @classmethod
    def create_random_initializer_key(cls, key_int: int) -> jax.Array:
        """Reconstruct a PRNGKey from an integer."""
        return jr.PRNGKey(key_int)


def build_equinox_MLP_from_config(eqx_config: EquinoxMLPConfig) -> eqx.Module:
    """
    Build an Equinox MLP from the provided configuration.
    """
    from equinox.nn import MLP

    return MLP(
        in_size=eqx_config.input_dimension,
        out_size=eqx_config.output_dimension,
        width_size=eqx_config.layer_width,
        depth=eqx_config.depth,
        activation=jax.jit(getattr(jax.nn, eqx_config.activation_name)),
        key=eqx_config.random_initializer_key,
    )


def build_equinox_MLP_from_config_and_load_weights(
    *, eqx_path: Path, eqx_config: EquinoxMLPConfig = None
) -> eqx.Module:
    """
    Build an Equinox MLP from the configuration (or load it from disk if not provided)
    and then load its weights from file.
    """
    if eqx_config is None:
        with open(Path(eqx_path, "config.json"), "r") as f:
            config_data = json.load(f)
        eqx_config = EquinoxMLPConfig(**config_data)
    mlp = build_equinox_MLP_from_config(eqx_config)
    return eqx.tree_deserialise_leaves(Path(eqx_path, "weights.eqx"), mlp)


def partion_eqx_nn_by_trainability(eqx_nn: eqx.Module):
    """
    Partition eqx Module parameters into trainable and non-trainable groups.
    """
    return eqx.partition(eqx_nn, eqx.is_array)


def _truncated_normal_initializer(weight: jax.Array, key: jr.PRNGKey) -> jax.Array:
    """
    Apply a truncated normal initialization to a weight array.

    This initializer computes a standard deviation based on the input dimension,
    and returns a new weight array with values drawn from a truncated normal distribution.
    """
    out_dim, in_dim = weight.shape
    stddev = math.sqrt(1 / in_dim)
    return stddev * jax.random.truncated_normal(key, shape=(out_dim, in_dim), lower=-2, upper=2)


def _get_nn_weights(model: eqx.Module) -> List[jax.Array]:
    """
    Retrieve the weight matrices from all linear layers in the model.
    """
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    return [x.weight for x in jax.tree_util.tree_leaves(model, is_leaf=is_linear) if is_linear(x)]


def _init_linear_layer_weights(model: eqx.Module, key: jr.PRNGKey) -> eqx.Module:
    """
    Reinitialize all linear layers in the model with a truncated normal initializer.

    Use this function if you wish to override the default initialization.
    """
    weights = _get_nn_weights(model)
    subkeys = jr.split(key, len(weights))
    new_weights = [_truncated_normal_initializer(w, sk) for w, sk in zip(weights, subkeys)]
    new_model = eqx.tree_at(_get_nn_weights, model, new_weights)
    return new_model


class EquinoxMLPWrapper(BaseModel, validate_assignment=False):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=False)

    params: Any
    static: Any
    eqx_config: EquinoxMLPConfig
    path: Path

    @property
    def nn(self):
        return eqx.combine(self.params, self.static)

    @model_serializer
    def serialize_everything(self, _info) -> Dict[str, Any]:
        self.path.mkdir(parents=True, exist_ok=True)
        print(f"saving eqx weights and config to {self.path}")
        eqx.tree_serialise_leaves(Path(self.path, "weights.eqx"), self.nn)
        with open(Path(self.path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.eqx_config.model_dump(), f, indent=4)
        return {"path": self.path, "eqx_config": self.eqx_config.model_dump()}

    @model_validator(mode="before")
    @classmethod
    def create_model(cls, values):
        path = values.get("path")
        if not path:
            raise ValueError("Missing required key: path")
        eqx_config = values.get("eqx_config")
        if not isinstance(eqx_config, EquinoxMLPConfig):
            # If eqx_config is not a dict, load it from file if load_from_file is specified.
            if not isinstance(eqx_config, dict) and values.get("load_from_file"):
                print("Loading config from file")
                with open(Path(path, "config.json")) as json_file:
                    eqx_config = json.load(json_file)
            else:
                print("Not loading config from file--using passed in eqx_config dict")
            eqx_config = EquinoxMLPConfig(**eqx_config)
        values["eqx_config"] = eqx_config

        if values.get("load_from_file"):
            print("Loading eqx MLP from file")
            eqx_nn = build_equinox_MLP_from_config_and_load_weights(eqx_config=eqx_config, eqx_path=values["path"])
        else:
            print("Building new eqx MLP from scratch")
            eqx_nn = build_equinox_MLP_from_config(eqx_config=eqx_config)
            eqx_nn = _init_linear_layer_weights(eqx_nn, eqx_config.random_initializer_key)
        values["params"], values["static"] = partion_eqx_nn_by_trainability(eqx_nn)
        return values
