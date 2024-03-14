import jax
import numpy as np
from opt_einsum import contract
from typing import Dict, Tuple

def embed_data_in_encoder_decoder_subspaces(
	input_output_data: Tuple[jax.Array],
	encoder_decoder_config_dict: Dict) -> Tuple[jax.Array]:
	################################################################################
	# Grab variables from config												   #
	################################################################################
	save_location = encoder_decoder_config_dict.get('save_location')
	encoder_decoder_dir = encoder_decoder_config_dict['encoder_decoder_dir']
	encoder_basis_filename = encoder_decoder_config_dict['encoder_basis_filename']
	encoder_cobasis_filename = encoder_decoder_config_dict['encoder_cobasis_filename']
	decoder_filename = encoder_decoder_config_dict.get('decoder_filename')
	reduced_zip_filename = 'mq_data_reduced.npz'

	################################################################################
	# Reduce the input data and save to file									   #
	################################################################################
	import numpy as np
	import jax.numpy as jnp
	if len(input_output_data) == 3:
		X, Y, dYdX = input_output_data
	else:
		X, Y = input_output_data
		dYdX = None
	reduced_X = \
		contract('dm,mr->dr', X, jnp.asarray(np.load(encoder_decoder_dir + encoder_basis_filename))) #400 x 4225	
	reduced_Y = (
		contract('du,ur->dr', Y, jnp.asarray(np.load(encoder_decoder_dir + decoder_filename))) 
		if decoder_filename 
		else Y)
	if save_location:
		np.save(save_location+"X_reduced.npy", reduced_X)
		print("Saved embedded training data files.")
		np.save(save_location+"Y_reduced.npy", reduced_Y)
	if dYdX is not None:
		#  Load the and project the Jacobian data with the encoder cobasis
		reduced_dYdX = \
			contract('dmu,mr->dur',
					dYdX,
					jnp.asarray(np.load(encoder_decoder_dir+encoder_cobasis_filename)))
		if save_location is not None:
			np.save(save_location+"J_reduced.npy", reduced_dYdX)
			np.savez(save_location+reduced_zip_filename, 
					m_data=reduced_X,
					q_data=reduced_Y,
					J_data = reduced_dYdX)
			print("Saved embedded training data files.")
		print('Successfully reduced the data.')
		return reduced_X, reduced_Y, reduced_dYdX
	else:
		if save_location is not None:
			np.savez(save_location+reduced_zip_filename,
					m_data=reduced_X,
					q_data=reduced_Y)		
			print("Saved embedded training data files.")
		print('Successfully reduced the data.')
		return reduced_X, reduced_Y
