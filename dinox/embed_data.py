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

from .data_utilities import __load_shaped_jax_array_direct_to_gpu, makedirs


# TODO: python dinox.embed_data --cli_args which uses the CLI arguments from reduce_data.py (deprecated -- to be removed)
def embed_data_in_encoder_decoder_subspaces(
    input_output_data: Tuple[jax.Array], encoder_decoder_config_dict: Dict
) -> Tuple[jax.Array]:
    # Disk IO side effects, stdout (printing) side effects exist
    # returns reduced X, fX, and possibly dfXdX: (if X, fX, dfXdX = input_output_data)

    # start_time = time.time()
    ################################################################################
    # Grab variables from config												   #
    ################################################################################
    save_dir = encoder_decoder_config_dict.get("save_dir")
    encoder_decoder_dir = encoder_decoder_config_dict["encoder_decoder_dir"]
    encoder_basis_filename = encoder_decoder_config_dict["encoder_basis_filename"]
    encoder_cobasis_filename = encoder_decoder_config_dict["encoder_cobasis_filename"]
    decoder_filename = encoder_decoder_config_dict.get("decoder_filename")
    reduced_zip_filename = "mq_data_reduced.npz"

    ################################################################################
    # Reduce the input data and save to file									   #
    ################################################################################
    if len(input_output_data) == 3:
        X, fX, dfXdX = input_output_data
    else:
        X, fX = input_output_data
        dfXdX = None
    reduced_X = contract(
        "nx,xr->nr",
        X,
        __load_shaped_jax_array_direct_to_gpu(
            encoder_decoder_dir + encoder_cobasis_filename, (X.shape[1], -1)
        ),
        backend="jax",
    )

    reduced_fX = (
        contract(
            "nf,fr->nr",
            fX,
            __load_shaped_jax_array_direct_to_gpu(
                encoder_decoder_dir + decoder_filename, (fX.shape[1], -1)
            ),
            backend="jax",
        )
        if decoder_filename
        else fX
    )
    if save_dir:
        makedirs(save_dir, exist_ok=True)
        jnp.save(save_dir + "X_reduced.npy", reduced_X)
        jnp.save(save_dir + "fX_reduced.npy", reduced_fX)
        print("Saved embedded training data files to disk.")
    if dfXdX is not None:
        #  Load the and project the Jacobian data with the encoder cobasis
        reduced_dfXdX = contract(
            "nxu,xr->nur",
            dfXdX,
            __load_shaped_jax_array_direct_to_gpu(
                encoder_decoder_dir + encoder_basis_filename, (X.shape[1], -1)
            ),
            backend="jax",
        )
        if save_dir:
            jnp.save(save_dir + "dfXdX_reduced.npy", reduced_dfXdX)
            jnp.savez(
                save_dir + reduced_zip_filename,
                X_data=reduced_X,
                fX_data=reduced_fX,
                dfXdX_data=reduced_dfXdX,
            )
            print("Saved zipped embedded training data file to disk.")
        print("Successfully reduced the data.")
        return reduced_X, reduced_fX, reduced_dfXdX
    else:
        if save_dir:
            jnp.savez(
                save_dir + reduced_zip_filename, X_data=reduced_X, dfXdX_data=reduced_fX
            )
            print("Saved zipped embedded training data file to disk.")
        print("Successfully reduced the data.")
        return reduced_X, reduced_fX


def main():
    # TODO: argparse move reduce_data.py CLI to here
    return 0


if __name__ == "__main__":
    sys.exit(main())
