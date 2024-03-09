

import sys
import numpy as np
from opt_einsum import contract

################################################################################
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', '--data_dir', type=str, default='../data/pointwise/', help="What directory for all data to be split")
parser.add_argument('-rb_dir', '--rb_dir', type=str, default='../reduced_bases/', help="Where are the reduced bases")
parser.add_argument('-input_basis', '--input_basis', type=str, default='kle', help="What type of input basis? Choose from [kle, as] ")
parser.add_argument('-output_basis', '--output_basis', type=str, default='pod', help="What type of input basis? Choose from [pod] ")
parser.add_argument('-J_data', '--J_data', type=int, default=1, help="Is there J data??? ")

args = parser.parse_args()


# Load the data:
file_to_reduce = 'mq_data.npz'
full_data = np.load(args.data_dir+file_to_reduce)
m_data = full_data['m_data']
u_data = full_data['q_data']
################################################################################
# Reduce the input data
# Load the projectors
if args.input_basis.lower() == 'kle':
	input_projector_file = 'KLE_projector.npy'
	input_basis_file = 'KLE_basis.npy'
elif args.input_basis.lower() == 'as':
	input_projector_file = 'AS_input_projector.npy'
	input_basis_file = 'AS_input_basis.npy'
else:
	raise

reduced_m_data = contract('mr,dm->dr',np.load(args.rb_dir + input_projector_file),m_data)

if args.J_data:	
	#  Load the Jacobian data
	reduced_J_data = contract('dmu,mr->dur',np.load(args.data_dir+'JstarPhi_data.npz')['JstarPhi_data'],np.load(args.rb_dir+input_basis_file))

if 'full_state' in args.data_dir:
	if args.output_basis.lower() == 'pod':
		output_projector_file = 'POD_projector.npy'
	else:
		raise
	reduced_u_data = contract('ur,du->dr',np.load(args.rb_dir + output_projector_file),u_data)
else:
	reduced_u_data = u_data

reduced_file_name = file_to_reduce.split('.npz')[0]+'_reduced.npz'

if args.J_data:
	np.savez(args.data_dir+reduced_file_name, reduced_m_data,q_data=reduced_u_data,\
												J_data = reduced_J_data)
	#save individual np files for new dino_training data approach (direct to GPU)
	np.save(args.data_dir+"m_data_reduced.npy", reduced_m_data)
	np.save(args.data_dir+"q_data_reduced.npy", reduced_u_data)
	np.save(args.data_dir+"J_data_reduced.npy", reduced_J_data)
else:
	np.savez(args.data_dir+reduced_file_name, m_data=reduced_m_data,q_data=reduced_u_data)
	np.save(args.data_dir+"m_data_reduced.npy", reduced_m_data)
	np.save(args.data_dir+"q_data_reduced.npy", reduced_u_data)
print('Successfully reduced the data.')