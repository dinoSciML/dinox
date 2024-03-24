import argparse
import sys  # temporary FIXME

import numpy as np
from opt_einsum import contract

sys.path.append("../../../")
import dinojax
from dinojax import embed_data_in_encoder_decoder_subspaces, load_data_from_disk

########################################################################################
# This file implements a CLI for running and saving the results of 					   #
# embed_training_data_in_encoder_decoder_subspaces to disk							   #
########################################################################################
parser = argparse.ArgumentParser()
parser.add_argument(
    "-training_data_dir",
    "--training_data_dir",
    type=str,
    default="../data/pointwise/",
    help="What directory for all data to be split",
)
parser.add_argument(
    "-rb_dir",
    "--rb_dir",
    type=str,
    default="../reduced_bases/",
    help="Where are the reduced bases",
)
parser.add_argument(
    "-encoder_basis",
    "--encoder_basis",
    type=str,
    default="kle",
    help="What type of input basis? Choose from [kle, as] ",
)
# parser.add_argument('-decoder_basis', '--decoder_basis', type=str, default='pod', help="What type of input basis? Choose from [pod] ")
# parser.add_argument('-J_data', '--J_data', type=int, default=1, help="Is there J data??? ")

args = parser.parse_args()

training_data_dir = args.training_data_dir
encoder_decoder_dir = args.rb_dir
# Load the encoders
if args.encoder_basis.lower() == "kle":
    encoder_cobasis_filename = "KLE_encoder_cobasis.npy"
    encoder_basis_filename = "KLE_encoder_basis.npy"
elif args.encoder_basis.lower() == "as":
    encoder_cobasis_filename = "AS_encoder_cobasis.npy"
    encoder_basis_filename = "AS_encoder_basis.npy"
else:
    raise

# #Load the decoder
if "full_state" in training_data_dir:  # FIXME, this isn't great
    if args.decoder_basis.lower() == "pod":
        decoder_filename = "POD_projector.npy"
    else:
        decoder_filename = None
else:
    decoder_filename = None

encoder_decoder_config_dict = {}
encoder_decoder_config_dict["save_location"] = training_data_dir
encoder_decoder_config_dict["encoder_decoder_dir"] = encoder_decoder_dir
encoder_decoder_config_dict["encoder_basis_filename"] = encoder_basis_filename
encoder_decoder_config_dict["encoder_cobasis_filename"] = encoder_cobasis_filename
encoder_decoder_config_dict["decoder_filename"] = decoder_filename

# #TEMPORARY, until I modify the data generator to save m_data, q_data, J_data separately:
mq_data = np.load(training_data_dir + "mq_data.npz")
N = mq_data["m_data"].shape[0]
# np.save(training_data_dir+'m_data.npy', mq_data['m_data'])
# np.save(training_data_dir+'q_data.npy', mq_data['q_data'])
# np.save(training_data_dir+"J_data.npy", np.load(training_data_dir+'JstarPhi_data.npz')['JstarPhi_data'])
del mq_data
# # END OF TEMPORARY

training_data_dict = {
    "data_dir": training_data_dir,
    "data_file_names": ("m_data.npy", "q_data.npy", "J_data.npy"),
    "N": N,
}
training_data = load_data_from_disk(training_data_dict)  # Involves Disk I/O

embed_data_in_encoder_decoder_subspaces(training_data, encoder_decoder_config_dict)
