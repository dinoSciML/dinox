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

from os import makedirs
from typing import Dict, Tuple

import jax
import numpy as np
# TODO: python dinox.embed_data --cli_args which uses the CLI arguments from reduce_data.py (deprecated -- to be removed)
import jax.numpy as jnp
from opt_einsum import contract

from ._data_utilities import __load_shaped_jax_array_direct_to_gpu, makedirs


def embed_data_in_encoder_decoder_subspaces(
    input_output_data: Tuple[jax.Array], encoder_decoder_config_dict: Dict
) -> Tuple[jax.Array]:
    """
    Processes the given input data using encoder and decoder subspaces defined in
    the configuration, reducing the data dimensionality, and optionally
    saves the results to disk.

    Parameters
    ----------
    input_output_data : Tuple[jax.Array]
        A tuple containing input data arrays X, fX, and optionally dfXdX (Jacobian matrix).
        The tuple can have two elements (X, fX) or three elements (X, fX, dfXdX).
    encoder_decoder_config_dict : Dict
        A dictionary containing configuration settings for the encoder/decoder,
        including file paths and directory settings for loading the basis matrices
        and saving the reduced data.

    Returns
    -------
    Tuple[jax.Array]
        A tuple of reduced data arrays: reduced X, reduced fX, and possibly reduced dfXdX.
        The tuple's contents depend on the input data provided.

    Side Effects
    ------------
    - Reads encoder and decoder basis matrices from disk.
    - Saves reduced data arrays to disk if specified in the configuration.
    - Prints status messages to stdout reflecting the progress and completion of data saving.

    Notes
    -----
    The function leverages tensor contraction operations to project input data onto the
    encoder and decoder subspaces, leading to data dimensionality reduction.
    The Jacobian data, if provided, is also projected using a specified encoder cobasis.
    """
    save_dir = encoder_decoder_config_dict.get("save_dir")
    encoder_basis_path = encoder_decoder_config_dict["encoder_basis_path"]
    encoder_cobasis_path = encoder_decoder_config_dict["encoder_cobasis_path"]
    decoder_cobasis_path = encoder_decoder_config_dict.get("decoder_cobasis_path")
    batched_encoding = encoder_decoder_config_dict["batchedEncoding"]
    reduced_zip_filename = "mq_data_reduced.npz" #TODO make this more general

    if len(input_output_data) == 3:
        X, fX, dfXdX = input_output_data
    else:
        X, fX = input_output_data
        dfXdX = None

    reduced_X = contract(
        "nx,xr->nr",
        X,
        __load_shaped_jax_array_direct_to_gpu(encoder_cobasis_path, (X.shape[1], -1)
        ),
        backend="jax",
    )

    reduced_fX = (
        contract(
            "nf,fr->nr",
            fX,
            __load_shaped_jax_array_direct_to_gpu(decoder_cobasis_path, (fX.shape[1], -1)
            ),
            backend="jax",
        )
        if decoder_cobasis_path
        else fX
    )

    if save_dir:
        makedirs(save_dir, exist_ok=True)
        jnp.save(save_dir + "/X_reduced.npy", reduced_X)
        jnp.save(save_dir + "/fX_reduced.npy", reduced_fX)
        print("Saved embedded training data files to disk.")
    
    if dfXdX is not None:
        #incrementally compute and transfer to CPU
        if batched_encoding:
            print("Batching the encoding of the Jacobians")
            total_len = len(dfXdX)
            batch_size = 50 #hard cocded
            start_idx, end_idx = 0, batch_size
            encoder = __load_shaped_jax_array_direct_to_gpu(encoder_basis_path, (X.shape[1], -1))
            reduced_dfXdX = []
            while start_idx < total_len:
                reduced_dfXdX_batch = \
                    contract("nxu,xr->nur", dfXdX[start_idx:end_idx], encoder, backend="jax")
                reduced_dfXdX.append(jax.device_put(reduced_dfXdX_batch, jax.devices("cpu")[0]))
                start_idx = end_idx
                end_idx = start_idx + batch_size
            reduced_dfXdX = jax.device_put(np.concatenate(reduced_dfXdX,axis = 0))
        else:
            reduced_dfXdX = contract(
                "nxu,xr->nur",
                dfXdX,
                __load_shaped_jax_array_direct_to_gpu(encoder_basis_path, (X.shape[1], -1)
                ),
                backend="jax",
            )
        if save_dir:
            jnp.save(save_dir + "/dfXdX_reduced.npy", reduced_dfXdX)
            jnp.savez(
                save_dir + "/" + reduced_zip_filename,
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
                save_dir + "/" + reduced_zip_filename,
                X_data=reduced_X,
                dfXdX_data=reduced_fX,
            )
            print("Saved zipped embedded training data file to disk.")
        print("Successfully reduced the data.")
        return reduced_X, reduced_fX


import sys


def main() -> int:
    """
    Main entry point for the script. Intended to handle command-line arguments and
    initiate the process of reducing data as specified in reduce_data.py.

    Returns
    -------
    int
        Exit status code. A return of 0 typically indicates success, any non-zero value
        indicates an error or abnormal completion.

    Notes
    -----
    This function is expected to integrate argparse for command-line interface handling,
    which is yet to be implemented. Once implemented, it will parse the command-line
    arguments to configure and control the data reduction process.

    TODO: Implement argparse to move reduce_data.py CLI functionality to here.
    """
    return 0


if __name__ == "__main__":
    sys.exit(main())
