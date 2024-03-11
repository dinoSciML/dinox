from opt_einsum import contract
from data_utilities import JacTrainingDataclass
from data_utilities import DinoTrainingDataclassType
from typing import Dict

def embed_data_in_encoder_decoder_subspaces(
	input_output_data: DinoTrainingDataclass
	encoder_decoder_config_dict: Dict) -> Tuple[]
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
	reduced_X = \
		contract('mr,dm->dr', np.load(encoder_decoder_dir + encoder_basis_filename), input_output_data.X)
		
	reduced_Y = (
		contract('ur,du->dr', np.load(encoder_decoder_dir + decoder_filename), input_output_data.Y) 
		if decoder_filename 
		else input_output_data.Y)
	if save_location:
		np.save(save_location"X_reduced.npy", reduced_X)
		np.save(save_location+"Y_reduced.npy", reduced_Y)
	if input_output_data.dYdX:
		#  Load the and project the Jacobian data with the encoder cobasis
		reduced_dYdX = \
			contract('dmu,mr->dur',
					 np.load(______+'JstarPhi_data.npz')['JstarPhi_data'],
					 np.load(save_location+encoder_cobasis_filename))
		if save_location
			np.save(save_location+"J_reduced.npy", reduced_dYdX)
			np.savez(save_location+reduced_zip_filename, 
					m_data=reduced_X,
					q_data=reduced_Y,
					J_data = reduced_dYdX)
		print('Successfully reduced the data.')
		return input_output_data.__class__(X=reduced_X, Y=reduced_Y, dYdX=reduced_dYdX)
	else:
		if save_location:
			np.savez(save_location+reduced_zip_filename,
					m_data=reduced_X,
					q_data=reduced_Y)		
		print('Successfully reduced the data.')
		return input_output_data.__class__(X=reduced_X, Y=reduced_Y)
