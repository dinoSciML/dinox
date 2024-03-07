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
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["KMP_WARNINGS"] = "FALSE" 
import numpy as np
import tensorflow as tf
if int(tf.__version__[0]) > 1:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

import time
import pickle


sys.path.append( os.environ.get('DINO_PATH'))
from dinojax import neural_network_settings, 

from argparse import ArgumentParser

# Arguments to be parsed from the command line execution
parser = ArgumentParser(add_help=True)
# Architectural parameters
parser.add_argument("-architecture", dest='architecture',required=False, default = 'rb_dense', help="architecture type: as_dense or generic_dense",type=str)
parser.add_argument("-input_basis", dest='input_basis',required=False, default = 'as',  help="input basis: as or kle",type=str)
parser.add_argument("-output_basis", dest='output_basis',required=False, default = 'jjt',  help="output basis: pod or jjt",type=str)
parser.add_argument("-fixed_input_rank", dest='fixed_input_rank',required=False, default = 200, help="rank for input of AS network",type=int)
parser.add_argument("-fixed_output_rank", dest='fixed_output_rank',required=False, default = 50, help="rank for output of AS network",type=int)
parser.add_argument("-truncation_dimension", dest='truncation_dimension',required=False, default = 200, help="truncation dimension for low rank networks",type=int)
parser.add_argument("-network_name", dest='network_name',required=True,  help="out name for the saved weights",type=str)

parser.add_argument("-data_dir", dest='data_dir',required=True,  help="out name for the saved weights",type=str)

# Optimization parameters
parser.add_argument("-total_epochs", dest='total_epochs',required=False, default = 100,  help="total epochs for training",type=int)

# Loss function parameters
parser.add_argument("-target_rank", dest='target_rank',required=False, default = 50,  help="target rank to be learned for Jacobian information",type=int)
parser.add_argument("-batch_rank", dest='batch_rank',required=False, default = 50,  help="batch rank parameter used in sketching of Jacobian information",type=int)
parser.add_argument("-l2_weight", dest='l2_weight',required=False, default = 1.,  help="weight for l2 term",type=float)
parser.add_argument("-h1_weight", dest='h1_weight',required=False, default = 1.,  help="weight for h1 term",type=float)

# Full J training
parser.add_argument("-train_full_jacobian", dest='train_full_jacobian',required=False, default = 1,  help="full J",type=int)

parser.add_argument("-train_data_size", dest='train_data_size',required=False, default = 9500,  help="training data size",type=int)
parser.add_argument("-test_data_size", dest='test_data_size',required=False, default = 500,  help="testing data size",type=int)

args = parser.parse_args()

problem_settings = {}


settings = neural_network_settings(problem_settings)

settings['data_dir'] = args.data_dir

settings['target_rank'] = args.target_rank
settings['batch_rank'] = args.batch_rank

settings['train_data_size'] = args.train_data_size
settings['test_data_size'] = args.test_data_size

settings['architecture'] = args.architecture
settings['depth'] = 6
settings['layer_rank'] = 50
settings['activation'] = 'gelu'

settings['fixed_input_rank'] = args.fixed_input_rank
settings['fixed_output_rank'] = args.fixed_output_rank
settings['truncation_dimension'] = args.truncation_dimension
settings['hidden_layer_dimensions'] = (settings['depth']-1)*[settings['truncation_dimension']]+[settings['fixed_output_rank']]

settings['input_basis'] = args.input_basis
settings['output_basis'] = 'pod'

settings['train_full_jacobian'] = args.train_full_jacobian
settings['opt_parameters']['train_full_jacobian'] = args.train_full_jacobian

settings['reduced_input_training'] = True
settings['reduced_output_training'] = False


if (settings['batch_rank'] == settings['target_rank']):
	settings['outer_epochs'] = 1
	settings['opt_parameters']['keras_epochs'] = args.total_epochs
else:
	settings['shuffle_every_epoch'] = True
	settings['outer_epochs'] = args.total_epochs
	settings['opt_parameters']['keras_epochs'] = 1
settings['opt_parameters']['keras_verbose'] = True


settings['opt_parameters']['loss_weights'] = [args.l2_weight,args.h1_weight]

steps_per_epoch = int(args.train_data_size/ settings['opt_parameters']['keras_batch_size'] )
total_steps = args.total_epochs*steps_per_epoch

# Custom optimizer
# lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(\
# 			boundaries=[int(0.25*total_steps),int(0.5*total_steps), int(0.75*total_steps)],\
# 	 		values=[0.001,0.0005,0.00025, 0.0001])


settings['opt_parameters']['optax_optimizer'] = optax.optimizers.Adam(learning_rate=lr_schedule)

settings['network_name'] = args.network_name

if args.l2_weight != 1.0:
	settings['network_name'] += 'l2_weight_'+str(args.l2_weight)


for loss_weight in settings['opt_parameters']['loss_weights']:
		assert loss_weight >= 0
################################################################################
# Set up training and testing data.


data_dir = settings['data_dir']


all_data = np.load(data_dir+'mq_data.npz')
JTPhi_data = np.load(data_dir+'JTPhi_data.npz')

m_data = all_data['m_data']
q_data = all_data['q_data']
# In this case \Phi = I_{d_Q} sp thos os the entire Jacobian
PhiTJ_data = np.transpose(JTPhi_data['JTPhi_data'], (0,2,1)) #REFACTOR WITHOUT HTE TRANSPOSE

m_test = m_data[-args.test_data_size:]
q_test = q_data[-args.test_data_size:]
PhiTJ_test = PhiTJ_data[-args.test_data_size:]

m_remaining = m_data[:-args.test_data_size]
q_remaining = q_data[:-args.test_data_size]
PhiTJ_remaining = PhiTJ_data[:-args.test_data_size]


n_remaining = m_remaining.shape[0]

n_train = args.train_data_size

if n_train > n_remaining:
	print('More data requested than available for training')
	n_train = n_remaining
	print('Using ',n_train,'instead')

m_train = m_remaining[:n_train]
q_train = q_remaining[:n_train]
PhiTJ_train = PhiTJ_remaining[:n_train]

# TODO: remove these dicts, only keep data, stored in jax arrays
# train_dict = {'m_data':m_train,'q_data':q_train, 'J_data':PhiTJ_train}
# test_dict = {'m_data': m_test, 'q_data': q_test, 'J_data':PhiTJ_test}


################################################################################
# Setup the reduced bases (if it applies)

projector_dict = {}
dQ = q_data[0].shape[0]
input_basis = np.load('../reduced_bases/AS_input_projector.npy')[:,:args.fixed_input_rank]
input_projector = np.load('../reduced_bases/R_AS_input_projector.npy')[:,:args.fixed_input_rank]

print('input_basis.shape = ',input_basis.shape)

print('input_projector.shape = ',input_projector.shape)

print('m_train.shape = ', train_dict['m_data'].shape)



check_orth = True
if check_orth:
	PsistarPsi = input_projector.T@input_basis
	print('||Psi^*Psi - I|| = ',np.linalg.norm(PsistarPsi - np.eye(PsistarPsi.shape[0])))


projector_dict['input'] = input_projector
projector_dict['output'] = np.eye(dQ)
projector_dict['last_layer_bias'] = np.mean(train_dict['q_data'],axis = 0)

#TODO: place reults form reduce_data in projector_dict

# projector_dict = setup_reduced_bases(settings,train_dict)
# Prune the data here if desired...
# if settings['reduced_input_training'] or settings['reduced_output_training']:
# 	assert settings['train_full_jacobian']
# 	# Need to pass the reduced input dimension in for network construction
# 	# The projector is assumed to have dims (dM,rM)
# 	assert len(projector_dict['input'].shape) == 2
# 	if settings['reduced_input_training']:
# 		settings['reduced_input_dim'] = projector_dict['input'].shape[1]
# 		print('reduced_input_dim = ',settings['reduced_input_dim'])
# 	if settings['reduced_output_training']:
# 		settings['reduced_output_dim'] = projector_dict['output'].shape[1]


# 	# Prune the data here:
# 	# First m->m_r = \Psi^*m = (RV_r)^Tm
	#TODO: just laod these directly from file (they've already been computed in reduce_data.py. todo later, merge reduce_data and cm_weighted_training)

	train_dict['m_full'] = 
	train_dict['m_data'] = #
	test_dict['m_full'] = 
	test_dict['m_data'] = 
# 	# Second reduce the Jacobian data using the basis (not projector)
	train_dict['J_full'] = 
	train_dict['J_data'] = 
	test_dict['J_full'] = 
	test_dict['J_data'] = 


#TODO: Reimplement below, setup_the_dino, train_dino, and restitch_and_postprocess

# def setup_the_dino


# ################################################################################
# # Set up the neural networks
# regressor = setup_the_dino(settings,train_dict,projector_dict,reduced_input_training = settings['reduced_input_training'],\
# 																reduced_output_training = settings['reduced_output_training'])

# regressor.summary()

# ################################################################################
# # Start the training
# r_logger = {}
# regressor = train_dino(settings, regressor,train_dict,test_dict, logger = r_logger)

# ################################################################################
# # Post-processing / re-stitching in the case of the reduced training.
# if settings['reduced_input_training'] or settings['reduced_output_training']:
# 	if True:
# 		train_dict['m_data'] = train_dict['m_full']
# 		train_dict['J_data'] = train_dict['J_full']
# 		test_dict['m_data'] = test_dict['m_full']
# 		test_dict['J_data'] = test_dict['J_full']
# 	final_logger = {}
# 	regressor = restitch_and_postprocess(regressor,settings,train_dict,test_dict,projector_dict,logger = final_logger)

# defrestitch_and_postprocess should also serialize the network weights etc

# logger = {'reduced':r_logger,'full': final_logger}

# logging_dir = 'logging/'
# os.makedirs(logging_dir,exist_ok = True)


# with open(logging_dir+args.network_name+'.pkl', 'wb+') as f:
#         pickle.dump(logger, f, pickle.HIGHEST_PROTOCOL)


