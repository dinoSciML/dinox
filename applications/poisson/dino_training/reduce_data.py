

import sys, os
import numpy as np

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

# Load the Jacobian data
if args.J_data:
	all_J_data = np.load(args.data_dir+'JstarPhi_data.npz')
	J_data = np.transpose(all_J_data['JstarPhi_data'],(0,2,1))

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

input_projector = np.load(args.rb_dir + input_projector_file)
reduced_m_data = np.einsum('mr,dm->dr',input_projector,m_data)

if args.J_data:
	input_basis = np.load(args.rb_dir+input_basis_file)
	reduced_J_data = np.einsum('mr,dum->dur',input_basis,J_data)


if 'full_state' in args.data_dir:
	if args.output_basis.lower() == 'pod':
		output_projector_file = 'POD_projector.npy'
	else:
		raise

		output_projector = np.load(args.rb_dir + output_projector_file)

	reduced_u_data = np.einsum('ur,du->dr',output_projector,u_data)

else:
	reduced_u_data = u_data



reduced_file_name = file_to_reduce.split('.npz')[0]+'_reduced.npz'

if args.J_data:
	np.savez(args.data_dir+reduced_file_name, m_data=reduced_m_data,q_data=reduced_u_data,\
												J_data = reduced_J_data)
else:
	np.savez(args.data_dir+reduced_file_name, m_data=reduced_m_data,q_data=reduced_u_data)

print('Successfully reduced the data.')