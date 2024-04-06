# This file is part of the dinox package
#
# dinox is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2 of the License, or any later version.
#
# dinox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# If not, see <http://www.gnu.org/licenses/>.
#
# Authors: Joshua Chen and Tom O'Leary-Roseberry
# Contact: joshuawchen@icloud.com | tom.olearyroseberry@utexas.edu

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from kvikio.numpy import LikeWrapper  # kvikio is optional dependency?
from opt_einsum import contract  # opt_einsum is a dependency

from .data_utilities import __load_shaped_jax_array_direct_to_gpu


# TODO: python dinox.embed_data --cli_args which uses the CLI arguments from reduce_data.py (deprecated -- to be removed)
def embed_data_in_encoder_decoder_subspaces(
    input_output_data: Tuple[jax.Array], encoder_decoder_config_dict: Dict
) -> Tuple[jax.Array]:
    # Disk IO side effects, stdout (printing) side effects exist
    # returns reduced X, Y, and possibly dYdX: (if X, Y, dYdX = input_output_data)

    # start_time = time.time()
    ################################################################################
    # Grab variables from config												   #
    ################################################################################
    save_location = encoder_decoder_config_dict.get("save_location")
    encoder_decoder_dir = encoder_decoder_config_dict["encoder_decoder_dir"]
    encoder_basis_filename = encoder_decoder_config_dict["encoder_basis_filename"]
    encoder_cobasis_filename = encoder_decoder_config_dict["encoder_cobasis_filename"]
    decoder_filename = encoder_decoder_config_dict.get("decoder_filename")
    reduced_zip_filename = "mq_data_reduced.npz"

    ################################################################################
    # Reduce the input data and save to file									   #
    ################################################################################
    if len(input_output_data) == 3:
        X, Y, dYdX = input_output_data
    else:
        X, Y = input_output_data
        dYdX = None
    reduced_X = contract(
        "nx,xr->nr",
        X,
        __load_shaped_jax_array_direct_to_gpu(
            encoder_decoder_dir + encoder_cobasis_filename, (X.shape[1], -1)
        ),
        backend="jax",
    )

    reduced_Y = (
        contract(
            "nu,ur->nr",
            Y,
            __load_shaped_jax_array_direct_to_gpu(
                encoder_decoder_dir + decoder_filename, (Y.shape[1], -1)
            ),
            backend="jax",
        )
        if decoder_filename
        else Y
    )
    if save_location:
        jnp.save(save_location + "X_reduced.npy", reduced_X)
        jnp.save(save_location + "Y_reduced.npy", reduced_Y)
        print("Saved embedded training data files to disk.")
    if dYdX is not None:
        #  Load the and project the Jacobian data with the encoder cobasis
        reduced_dYdX = contract(
            "nxu,xr->nur",
            dYdX,
            __load_shaped_jax_array_direct_to_gpu(
                encoder_decoder_dir + encoder_basis_filename, (X.shape[1], -1)
            ),
            backend="jax",
        )
        if save_location is not None:
            jnp.save(save_location + "J_reduced.npy", reduced_dYdX)
            jnp.savez(
                save_location + reduced_zip_filename,
                m_data=reduced_X,
                q_data=reduced_Y,
                J_data=reduced_dYdX,
            )
            print("Saved zipped embedded training data file to disk.")
        print("Successfully reduced the data.")
        return reduced_X, reduced_Y, reduced_dYdX
    else:
        if save_location is not None:
            jnp.savez(
                save_location + reduced_zip_filename, m_data=reduced_X, q_data=reduced_Y
            )
            print("Saved zipped embedded training data file to disk.")
        print("Successfully reduced the data.")
        return reduced_X, reduced_Y


def main():
    # TODO: argparse move reduce_data.py CLI to here
    return 0


if __name__ == "__main__":
    sys.exit(main())
