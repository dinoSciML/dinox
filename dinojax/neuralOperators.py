from typing import Sequence
from flax import linen as nn
import jax
import jax.numpy as jnp

# class GenericDense(nn.Module):
#     layer_widths: Sequence[int]
#     output_size: int 
#     activation: str
#     output_bias : bool = True

#     def setup(self):
#         assert self.activation in ["softplus", "tanh", "relu", "linear","gelu"]
#         self.hidden_layers = [nn.Dense(width) for width in self.layer_widths]
#         self.final_layer = nn.Dense(self.output_size, use_bias=self.output_bias)

#     def __call__(self, x):
#         for i, layer in enumerate(self.hidden_layers):
#             x = layer(x)
#             if self.activation == "softplus":
#                 x = nn.softplus(x)
#             elif self.activation == "tanh":
#                 x = nn.tanh(x)
#             elif self.activation == "relu":
#                 x = nn.relu(x)
#             elif self.activation == "gelu":
#                 x = nn.gelu(x)
#         x = self.final_layer(x)

#         return x