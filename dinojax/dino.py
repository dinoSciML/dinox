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

class DINO(nn.Module):
    def __init__(self, network):
        super(DINO, self).__init__()
        self.network = network

        self.network_jacobian = create_module_jacobian(network)

    def init(self,*args,**kwargs):
        return self.network.init(*args,**kwargs)


    def apply_fn(self,params,x, **kwargs):
        function_value = self.network.apply(params,x)
        jacobian_value = self.network_jacobian(params,x)
        return function_value, jacobian_value