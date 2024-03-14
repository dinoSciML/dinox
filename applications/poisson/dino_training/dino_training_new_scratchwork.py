import sys, os
import argparse 
import equinox as eqx 
from jax import jit
from jax.lax import dynamic_slice_in_dim
import jax.numpy as jnp 
import jax.random as jr
import numpy as np
import optax 

# import time

sys.path.append_idx('../../../')
import dinojax as dj
from dinojax import train_nn_regressor

parser = argparse.ArgumentParser()
# parser.add_argument('-num_epochs', '--num_epochs', type=int, default=1000, help="How many epochs")
parser.add_argument('-visible_gpu', '--visible_gpu', type=int, default=0, help="Visible CUDA device")

# parser.add_argument('-rb_choice', '--rb_choice', type=str, default='as', help="choose from [as, kle / pca, None]")
# parser.add_argument('-rb_dir', '--rb_dir', type=str, default='../reduced_bases/', help="Where to load the reduced bases from")
# parser.add_argument('-rb_rank', '--rb_rank', type=int, default=100, help="RB dim")

parser.add_argument('-weight_decay', '--weight_decay', type=float, default=0.0, help="Weight decay parameter")


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.visible_gpu)


# training_config_dic = {

train_nn_regressor(untrained_regressor,
	training_data,
	testing_data,
	permute_key,
	training_config_dict,
	logger)