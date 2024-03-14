# This file is part of the dino package
#
# dino is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2 of the License, or any later version.
#
# dino is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# If not, see <http://www.gnu.org/licenses/>.
#
# Author: Joshua Chen and Tom O'Leary-Roseberry
# Contact: joshuawchen@icloud.com | tom.olearyroseberry@utexas.edu

import sys
import numpy as np
import jax.random as jr
from argparse import ArgumentParser, BooleanOptionalAction

# sys.path.append( os.environ.get('DINOJAX_PATH'))
sys.path.append('../../../') #temporary

from dinojax import train_dino_in_embedding_space

################################################################################
# Define CLI arguments 
################################################################################
parser = ArgumentParser(add_help=True)

# Random seed parameter
parser.add_argument('-run_seed', '--run_seed', type=int, default=7, help="Seed for NN initialization/ data shuffling / initialization")

# Neural Network Architecture parameters
parser.add_argument("-architecture", dest='architecture',required=False, default = 'generic_dense', help="architecture type: as_dense or generic_dense",type=str)
# parser.add_argument("-decoder", dest='decoder',required=False, default = 'jjt',  help="output basis: pod or jjt",type=str)
parser.add_argument("-fixed_input_rank", dest='fixed_input_rank',required=False, default = 200, help="rank for input of AS network",type=int)
parser.add_argument("-fixed_output_rank", dest='fixed_output_rank',required=False, default = 50, help="rank for output of AS network",type=int)
parser.add_argument("-truncation_dimension", dest='truncation_dimension',required=False, default = 200, help="truncation dimension for low rank networks",type=int)

# Neural Network Serialization parameters
parser.add_argument("-network_name", dest='network_name',required=True,  help="out name for the saved weights",type=str)

# Data (Directory Location/Training) parameters
parser.add_argument("-data_dir", dest='data_dir',required=True,  help="Directory where training data lies",type=str)
parser.add_argument("-train_data_size", dest='train_data_size',required=False, default = 4500,  help="training data size",type=int)
parser.add_argument("-test_data_size", dest='test_data_size',required=False, default = 500,  help="testing data size",type=int)

# Optimization parameters
parser.add_argument("-optax_optimizer", dest='optax_optimizer',required=False, default = 'adam',  help="Name of the optax optimizer to use",type=str)
parser.add_argument("-total_epochs", dest='total_epochs',required=False, default = 1000,  help="total epochs for training",type=int)
parser.add_argument('-batch_size', dest='batch_size', type = int, default = 20, help = 'gradient batch size')
parser.add_argument('-step_size', '--step_size', type=float, default=1e-3, help="What step size or 'learning rate'?")

# Loss function parameters
parser.add_argument("-l2_weight", dest='l2_weight',required=False, default = 1.,  help="weight for l2 term",type=float)

parser.add_argument("-h1_weight", dest='h1_weight',required=False, default = 1.,  help="weight for h1 term",type=float)


# Encoder/Decoder parameters
parser.add_argument('-rb_dir', '--rb_dir', type=str, default='../reduced_bases/', help="Where are the reduced bases")
parser.add_argument('-encoder_basis', '--encoder_basis', required=False, type=str, default='as', help="What type of input basis? Choose from [kle, as] ")
parser.add_argument('-decoder_basis', '--decoder_basis', type=str, default='pod', help="What type of input basis? Choose from [pod] ")
parser.add_argument('-save_embedded_data', '--save_embedded_data', help="Should we save the embedded training data to disk or just use it in training without saving to disk. WIthout this flag, defaults to false", default=False, action=BooleanOptionalAction)
# parser.add_argument('-J_data', '--J_data', type=int, default=1, help="Is there J data??? ")

args = parser.parse_args()

################################################################################
# Parse arguments and place them in a config dictionary 	    			   #
################################################################################
args = parser.parse_args()
problem_config_dict = {} #is this necessary

config_dict = {'nn':{},'data':{},'training':{},'encoder_decoder':{},'network_serialization':{}}

config_dict['forward_problem'] = problem_config_dict

# Neural Network Architecture parameters
config_dict['nn']['architecture'] = args.architecture
config_dict['nn']['depth'] = 6 
#TODO: CHECK ON THIS, as a functio nof DIMENSION REDUCTION PARMETERS!
config_dict['nn']['layer_width'] = 2*50 #args.rb_rank 
# config_dict['nn']['layer_rank'] = 50 #nn_width = 2*args.rb_rank?
config_dict['nn']['activation'] = 'gelu'
# config_dict['hidden_layer_dimensions'] = (config_dict['depth']-1)*[config_dict['truncation_dimension']]+[config_dict['fixed_output_rank']]

#Encoder/Decoder parameters
config_dict['encoder_decoder']['encode'] = True
config_dict['encoder_decoder']['decode'] = False
config_dict['encoder_decoder']['encoder'] = args.encoder_basis
config_dict['encoder_decoder']['decoder'] = 'pod' #ignored for now, decode is False
encoder_decoder_dir = f'{args.data_dir}reduced_bases/' if args.rb_dir=='' else args.rb_dir
if args.encoder_basis.lower() == 'kle':
	encoder_cobasis_filename = 'KLE_cobasis.npy'
	encoder_basis_filename = 'KLE_basis.npy'
elif args.encoder_basis.lower() == 'as':
	encoder_cobasis_filename = 'AS_encoder_cobasis.npy'
	encoder_basis_filename = 'AS_encoder_basis.npy'
else:
	raise
if 'full_state' in args.data_dir:
	if args.decoder_basis.lower() == 'pod':
		decoder_filename = 'POD_projector.npy'
	else:
		decoder_filename = None
else:
	decoder_filename = None
config_dict['encoder_decoder']['save_location'] = \
	args.data_dir if args.save_embedded_data else None
config_dict['encoder_decoder']['encoder_decoder_dir'] = encoder_decoder_dir
config_dict['encoder_decoder']['encoder_basis_filename'] = encoder_basis_filename
config_dict['encoder_decoder']['encoder_cobasis_filename'] = encoder_cobasis_filename
config_dict['encoder_decoder']['decoder_filename'] = decoder_filename
# config_dict['encoder_decoder']['reduced_data_filenames'] = ('X_reduced.npy','Y_reduced.npy','J_reduced.npy') #these files may not exist

# Data (Directory Location/Training) parameters
config_dict['data']['data_dir'] = args.data_dir
config_dict['data']['train_data_size'] = args.train_data_size
config_dict['data']['test_data_size'] = args.test_data_size
if config_dict['encoder_decoder']['encode'] or config_dict['encoder_decoder']['decode']:
	config_dict['data']['data_filenames'] = \
		('m_data.npy','q_data.npy','J_data.npy')
else:
	config_dict['data']['data_filenames'] = \
		('m_data.npy','q_data.npy','J_data.npy')

# Optimization parameters
config_dict['training']['step_size'] = args.step_size
config_dict['training']['batch_size'] = args.batch_size
config_dict['training']['optax_optimizer'] = args.optax_optimizer
config_dict['training']['optax_epochs'] = args.total_epochs
config_dict['training']['shuffle_every_epoch'] = True
loss_weights = [args.l2_weight,args.h1_weight]
for loss_weight in loss_weights:
	assert loss_weight >= 0
config_dict['training']['loss_weights'] = loss_weights

# config_dict['truncation_dimension'] = args.truncation_dimension

# Deserializing / Serializing Neural Network settings
config_dict['network_serialization']['network_name'] = args.network_name
config_dict['network_serialization']['save_weights'] = True
config_dict['network_serialization']['weights_dir'] = 'trained_weights/'
# config_dict['network_serialization']['initial_guess_path'] = 

if args.l2_weight != 1.0:
	config_dict['network_serialization']['network_name'] += f'l2_weight_{args.l2_weight}_seed_{args.run_seed}'
train_dino_in_embedding_space(model_key = jr.PRNGKey(args.run_seed), 
							  embedded_training_config_dict=config_dict)

#2 digits in the F and 1 digit in the Jacobian
