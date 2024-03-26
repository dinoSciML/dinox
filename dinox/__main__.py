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

import os
###################################################################
# To run from CLI: python -m dinox --<cli_args_name> <cli_arg>
###################################################################
import sys
from argparse import ArgumentParser, BooleanOptionalAction

from .train import sub_dict, train_dino_in_embedding_space


def main() -> int:
    """"""
    ################################################################################
    # Define CLI arguments
    ################################################################################
    cli = ArgumentParser(add_help=True)

    # Random seed parameter
    cli.add_argument(
        "-run_seed",
        "--run_seed",
        type=int,
        default=777,
        help="Seed for NN initialization/ data shuffling / initialization",
    )

    # Neural Network Architecture parameters
    cli.add_argument(
        "-architecture",
        dest="architecture",
        required=False,
        default="generic_dense",
        help="architecture type: as_dense or generic_dense",
        type=str,
    )
    cli.add_argument(
        "-activation",
        dest="activation",
        required=False,
        default="gelu",
        help="activation type: e.g. 'gelu', 'relu'",
        type=str,
    )
    cli.add_argument(
        "-depth",
        dest="depth",
        required=False,
        default=8,  # 8 #6
        help="NN # of layers (depth): e.g. 8",
        type=int,
    )
    # cli.add_argument("-decoder", dest='decoder',required=False, default = 'jjt',  help="output basis: pod or jjt",type=str)
    cli.add_argument(
        "-fixed_input_rank",
        dest="fixed_input_rank",
        required=False,
        default=200,
        help="rank for input of AS network",
        type=int,
    )
    cli.add_argument(
        "-fixed_output_rank",
        dest="fixed_output_rank",
        required=False,
        default=50,
        help="rank for output of AS network",
        type=int,
    )
    cli.add_argument(
        "-truncation_dimension",
        dest="truncation_dimension",
        required=False,
        default=200,
        help="truncation dimension for low rank networks",
        type=int,
    )

    # Neural Network Serialization parameters
    cli.add_argument(
        "-network_name",
        dest="network_name",
        required=True,
        help="out name for the saved weights",
        type=str,
    )

    # Data (Directory Location/Training) parameters
    cli.add_argument(
        "-data_dir",
        dest="data_dir",
        required=True,
        help="Directory where training data lies",
        type=str,
    )
    cli.add_argument(
        "-train_data_size",
        dest="train_data_size",
        required=False,
        default=2000,
        help="training data size",
        type=int,
    )
    cli.add_argument(
        "-test_data_size",
        dest="test_data_size",
        required=False,
        default=3000,
        help="testing data size",
        type=int,
    )

    # Optimization parameters
    cli.add_argument(
        "-optax_optimizer",
        dest="optax_optimizer",
        required=False,
        default="adam",
        help="Name of the optax optimizer to use",
        type=str,
    )
    cli.add_argument(
        "-n_epochs",
        dest="n_epochs",
        required=False,
        default=10000,
        help="number of epochs for training",
        type=int,
    )
    cli.add_argument(
        "-batch_size", dest="batch_size", type=int, default=100, help="batch size"
    )
    cli.add_argument(
        "-step_size",
        "--step_size",
        type=float,
        default=1e-4,
        help="What step size or 'learning rate'?",
    )

    # Loss function parameters
    cli.add_argument(
        "-l2_weight",
        dest="l2_weight",
        required=False,
        default=1.0,
        help="weight for l2 term",
        type=float,
    )
    cli.add_argument(
        "-h1_weight",
        dest="h1_weight",
        required=False,
        default=1.0,
        help="weight for h1 term",
        type=float,
    )

    # Encoder/Decoder parameters
    cli.add_argument(
        "-rb_dir",
        "--rb_dir",
        type=str,
        default="",
        help="Where are the reduced bases",
    )
    cli.add_argument(
        "-encoder_basis",
        "--encoder_basis",
        required=False,
        type=str,
        default="as",
        help="What type of input basis? Choose from [kle, as] ",
    )
    cli.add_argument(
        "-decoder_basis",
        "--decoder_basis",
        type=str,
        default="pod",
        help="What type of input basis? Choose from [pod] ",
    )
    cli.add_argument(
        "-save_embedded_data",
        "--save_embedded_data",
        help="Should we save the embedded training data to disk or just use it in training without saving to disk. WIthout this flag, defaults to false",
        default=False,
        action=BooleanOptionalAction,
    )
    # cli.add_argument('-J_data', '--J_data', type=int, default=1, help="Is there J data??? ")
    ################################################################################
    # Parse arguments and place them in a heirarchical config dictionary 	       #
    ################################################################################
    cli_args = vars(cli.parse_args())

    ###################################################################################
    # Define the keys for each configuration dict (*_keys is a required naming
    # convention here, since we define the configuration dict names by *
    ##################################################################################
    # right now this is only for dense.
    nn_keys = ("architecture", "depth", "activation")  #'layer_width',
    data_keys = (
        "data_dir",
        "train_data_size",
        "test_data_size",
    )  #'data_filenames'
    training_keys = ("step_size", "batch_size", "optax_optimizer", "n_epochs")
    network_serialization_keys = ("network_name",)
    config_dict = {
        k.removesuffix("_keys"): sub_dict(super_dict=cli_args, keys=v)
        for k, v in locals().items()
        if k.endswith("_keys")
    }
    print(config_dict)
    # ESS, max weight, 3rd order moment, k-fold cross validation

    # problem_config_dict = {} #is this necessary
    # config_dict['forward_problem'] = problem_config_dict

    # TEMPORARY, modify additionally each configuraiton dict. THese really should be
    # CLI commands of some sort, or part of the default associated with CLI params
    # e.g. nn['generic_dense'] default.

    # Neural Network Architecture parameters
    # TODO: CHECK ON THIS, as a functio nof DIMENSION REDUCTION PARMETERS!
    config_dict["nn"]["layer_width"] = 2 * 50  # args.rb_rank
    # config_dict['nn']['layer_rank'] = 50 #nn_width = 2*args.rb_rank?
    # config_dict['hidden_layer_dimensions'] = (config_dict['depth']-1)*[config_dict['truncation_dimension']]+[config_dict['fixed_output_rank']]

    # Encoder/Decoder parameters
    config_dict["encoder_decoder"] = {}
    config_dict["encoder_decoder"]["encode"] = True
    config_dict["encoder_decoder"]["decode"] = False
    config_dict["encoder_decoder"]["encoder"] = cli_args["encoder_basis"]
    config_dict["encoder_decoder"][
        "decoder"
    ] = "pod"  # ignored for now, decode is False
    encoder_decoder_dir = (
        f"{cli_args['data_dir']}reduced_bases/"
        if cli_args["rb_dir"] == ""
        else cli_args["rb_dir"]
    )
    encoder_basis = config_dict["encoder_decoder"]["encoder"].upper()
    decoder_basis = cli_args["decoder_basis"].upper()
    assert encoder_basis in ("AS", "KLE")
    if (
        "full_state" in cli_args["data_dir"]
        and config_dict["encoder_decoder"]["decode"]
    ):
        decoder_filename = f"{decoder_basis}_projector.npy"
    else:
        decoder_filename = None
    # save_location means embedded_data_save_dir
    config_dict["encoder_decoder"]["save_location"] = (
        cli_args["data_dir"] if cli_args["save_embedded_data"] else None
    )
    config_dict["encoder_decoder"]["encoder_decoder_dir"] = encoder_decoder_dir
    config_dict["encoder_decoder"][
        "encoder_basis_filename"
    ] = f"{encoder_basis}_encoder_basis.npy"
    config_dict["encoder_decoder"][
        "encoder_cobasis_filename"
    ] = f"{encoder_basis}_encoder_cobasis.npy"
    config_dict["encoder_decoder"]["decoder_filename"] = decoder_filename
    # config_dict['encoder_decoder']['reduced_data_filenames'] = ('X_reduced.npy','Y_reduced.npy','J_reduced.npy') #these files may not exist

    # Data (Directory Location/Training) parameters
    if (
        config_dict["encoder_decoder"]["encode"]
        or config_dict["encoder_decoder"]["decode"]
    ):
        config_dict["data"]["data_filenames"] = (
            "m_data.npy",
            "q_data.npy",
            "J_data.npy",
        )
    else:
        config_dict["data"]["data_filenames"] = (
            "m_data.npy",
            "q_data.npy",
            "J_data.npy",
        )

    # Optimization parameters
    loss_weights = [cli_args["l2_weight"], cli_args["h1_weight"]]
    for loss_weight in loss_weights:
        assert loss_weight >= 0
    config_dict["training"]["loss_weights"] = loss_weights

    # config_dict['truncation_dimension'] = args.truncation_dimension

    # Deserializing / Serializing Neural Network settings
    config_dict["network_serialization"]["save_weights"] = True  # make a CLI
    config_dict["network_serialization"][
        "weights_dir"
    ] = "trained_weights/"  # make a CLI
    # config_dict['network_serialization']['initial_guess_path'] =

    random_seed = cli_args["run_seed"]
    if cli_args["l2_weight"] != 1.0:
        config_dict["network_serialization"][
            "network_name"
        ] += f"l2_weight_{cli_args['l2_weight']}_seed_{random_seed}"

    train_dino_in_embedding_space(
        random_seed=random_seed, embedded_training_config_dict=config_dict
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
