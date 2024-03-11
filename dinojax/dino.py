import jax
import jax.numpy as jnp
from flax import linen as nn

def create_module_jacobian(module, mode="forward"):
    def forward_pass(params, x):
        return module.apply(params, jnp.expand_dims(x, 0))[0]
    
    if mode == "forward":
        jac = jax.jacfwd(forward_pass, argnums=1)
    elif mode == "reverse":
        jac = jax.jacrev(forward_pass, argnums=1)
    else:
        raise ValueError("Incorrect AD mode")

    return jax.jit(jax.vmap(jac, (None, 0)))

class DINO(nn.Module): #flax dino
    def __init__(self, network):
        super(DINO, self).__init__()
        self.network = network

        self.network_jacobian = jax.jit(create_module_jacobian(network))

    def init(self,*args,**kwargs):
        return self.network.init(*args,**kwargs)

    def apply_fn(self,params,x, **kwargs):
        function_value = jax.jit(self.network.apply)(params,x)
        jacobian_value = self.network_jacobian(params,x)
        return function_value, jacobian_value

import jax.nn

def GenericDenseFactory(*, layer_width, depth, input_size, output_size, activation = 'gelu'):
    "DOCUMENT ME"
    return eqx.nn.MLP(in_size = input_size, out_size=output_size, width_size=width_size, depth=depth, activation=jax.nn.__dict__[activation], key=model_key)

#TODO: implement other Neural Networks

def instantiate_uninitialized_nn(nn_config_dict):
    "DOCUMENT ME"
    if nn_config_dict['architecture'] is 'generic_dense':
        relevant_keys = ['layer_width', 'depth', 'input_size', 'output_size', 'activation']
        return GenericDenseFactory(**{key: nn_config_dict[key] for key in relevant_keys})
    else:
        raise("not implemented")

def instantiate_nn(model_seed, nn_config_dict):
    """
    This function sets up the dino network for training
    """
    ################################################################################
    # Set up the neural network
    ################################################################################
    eqx_nn_regressor = instantiate_uninitialized_nn(nn_config_dict)

    ################################################################################
    # Random seed
    ################################################################################
    permute_key, model_key = jr.split(jr.PRNGKey(model_seed), 2)

    ################################################################################
    # Load equinox NN parameter checkpoint (as an initial guess for optimization)
    # into equinox NN model (pytrees)
    ################################################################################
    jax_serialized_params_path = nn_config_dict.get('initial_guess_path')
    if jax_serialized_params_path:
        assert os.path.isfile(jax_serialized_params_path), 'Trained weights may not exist as specified: '+str(jax_serialized_params_path)
        eqx_nn_regressor = eqx.tree_deserialise_leaves(
            jax_serialized_params_path, 
            regrequinox_nn_regressoressor)
    else:
        pass
        #TODO: intialize jax neural network wieghts with random weights!!!!
        # use model_key to initialize model
        # #INITIALIZE WITH DEFAULT WEIGHT APPROACH

        # @jit
        # def initialize(params_rng):
        #     init_rngs = {'params': params_rng}
        #     input_shape = (1, dM)
        #     variables = network.init(init_rngs, jnp.ones(input_shape, jnp.float32))
        #     return variables

        # eqx_nn_regressor.init(model_key)

    return eqx_nn_regressor