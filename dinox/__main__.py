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

###################################################################
# To run from CLI: python -m dinox --<cli_args_name> <cli_arg>
###################################################################
import sys
from argparse import ArgumentParser, BooleanOptionalAction

from .train import sub_dict, train_dino_in_embedding_space
from .data_utilities import makedirs, save_to_pickle
from pathlib import Path
import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)


# TODO: add decorator for saving the returns of functions... rather than dealing with
# TODO: switch from argparse to click, use @click.option()
# TODO: use a logger for print statementes ('verbose=True') rather than printing
def main() -> int:
    """"""
    ################################################################################
    # Define CLI arguments
    ################################################################################
    cli = ArgumentParser(add_help=True)

    # Random seed parameter
    cli.add_argument(
        "-run_seed",
        type=int,
        default=777,
        help="Seed for NN initialization/ data shuffling / initialization",
    )

    # Neural Network Architecture parameters
    cli.add_argument(
        "-architecture",
        type=str,
        default="generic_dense",
        help="architecture type: generic_dense",
    )
    cli.add_argument(
        "-activation",
        type=str,
        default="gelu",
        help="activation type: e.g. 'gelu', 'relu'",
    )
    cli.add_argument(
        "-depth",
        type=int,
        default=8,  # 6
        help="NN # of layers (depth): e.g. 8",
    )
    # cli.add_argument("-decoder", dest='decoder',required=False, default = 'jjt',  help="output basis: pod or jjt",type=str)
    # cli.add_argument(
    #     "-fixed_input_rank",
    #     dest="fixed_input_rank",
    #     required=False,
    #     default=200,
    #     help="rank for input of AS network",
    #     type=int,
    # )
    # cli.add_argument(
    #     "-fixed_output_rank",
    #     dest="fixed_output_rank",
    #     required=False,
    #     default=50,
    #     help="rank for output of AS network",
    #     type=int,
    # )
    # cli.add_argument( #reintroduce if we need the dimension of the problem to be lower than the number of bases
    # given by AS
    #     "-truncation_dimension",
    #     dest="truncation_dimension",
    #     required=False,
    #     default=200,
    #     help="truncation dimension for low rank networks",
    #     type=int,
    # )

    # Neural Network Serialization parameters
    cli.add_argument(
        "-nn_save_name",
        type=str,
        required=True,
        help="out name for the serialized NN weights",
    )
    cli.add_argument(
        "-loss_results_dir",
        type=str,
        default="training_metrics/",
        help="Where in `problem_dir` are the training/generalization error results from training?",
    )

    # Data (Directory Location/Training) parameters
    cli.add_argument(
        "-problem_dir",
        type=str,
        required=True,
        help="Where is the problem located?",
    )
    cli.add_argument(
        "-samples_dir",
        type=str,
        default="samples/",
        help="Where in `problem_dir` are the training samples?",
    )
    cli.add_argument(
        "-train_data_size", #n_train
        type=int,
        required=True,
        help="Training data size, e.g. 2000",
    )
    cli.add_argument(
        "-test_data_size", #n_test
        type=int,
        default=5000,
        help="Testing data size, e.g. 5000",
    )

    # Optimization parameters
    cli.add_argument("-batch_size", type=int, default=25, help="Training batch size")
    cli.add_argument("-optax_optimizer", type=str, default="adam", help="Optax optimizer choice")
    cli.add_argument( "-n_epochs", type=int, default=500, help="# training epochs")
    cli.add_argument("-step_size", type=float, default=1e-3, help="What step size/learning rate?")

    # Loss function parameters
    cli.add_argument(
        "-l2_weight",
        type=float,
        default=1.0,
        help="weight for l2 term",
    )
    cli.add_argument(
        "-h1_weight",
        type=float,
        default=1.0,
        help="weight for h1 term",
    )

    # Encoder/Decoder parameters
    cli.add_argument(
        "-encoder_dir",
        type=str,
        default="encoder/",
        help="What subdirectory of `problem_dir` are the encoder bases/cobases located in",
    )
    cli.add_argument(
        "-encoder",
        type=str,
        default="as",
        help="What type of encoder? Choose from [kle, as] ",
    )
    cli.add_argument(
        "-decoder",
        type=str,
        default="pod",
        help="What type of decoder? Choose from [pod] ",
    )
    cli.add_argument(
        "-save_embedded_data",
        help="Save embedded training data to disk after encoder embeds it?",
        default=False,
        action=BooleanOptionalAction,
    )
    cli.add_argument(
        "-save_embedded_dir",
        type=str,
        default="embedded_samples/",
        help="If `save_embedded_data`, where in `problem_dir` should we save the embedded training data?",
    )
    ################################################################################
    # Parse arguments and place them in a heirarchical config dictionary 	       #
    ################################################################################
    cli_args = vars(cli.parse_args())

    ###################################################################################
    # Define the keys for each configuration dict (*_keys is a required naming
    # convention here, since we define the configuration dict names by *
    ##################################################################################
    # right now this is only for dense, TODO: generalize
    nn_keys = ("architecture", "depth", "activation")
    data_keys = ("samples_dir", "train_data_size", "test_data_size")  #'data_filenames'
    training_keys = ("step_size", "batch_size", "optax_optimizer", "n_epochs")
    network_serialization_keys = ("nn_save_name","loss_results_dir")
    encoder_decoder_keys = (
        "encoder_dir",
        "encoder",
        "decoder",
        "save_embedded_data",
        "save_embedded_dir",
    )
    # This line has to be called in the same scope as the keys defined above
    config_dict = {
        k.removesuffix("_keys"): sub_dict(super_dict=cli_args, keys=v)
        for k, v in locals().items()
        if k.endswith("_keys")
    }
    print(config_dict)
    # ESS, max weight,  k-fold cross validation

    problem_dir = cli_args["problem_dir"]
    # problem_config_dict = {} #is this necessary
    # config_dict['forward_problem'] = problem_config_dict

    # TEMPORARY, modify additionally each configuraiton dict. THese really should be
    # CLI commands of some sort, or part of the default associated with CLI params
    # e.g. nn['generic_dense'] default.

    # Encoder/Decoder parameters
    encodec_dict = config_dict["encoder_decoder"]
    encoder_basis = encodec_dict["encoder"].upper()
    decoder_basis = encodec_dict["decoder"].upper()
    assert encoder_basis in ("AS", "KLE")

    config_dict["encoder_decoder"]["encode"] = True
    config_dict["encoder_decoder"]["decode"] = False
    config_dict["encoder_decoder"][
        "encoder_decoder_dir"
    ] = f"{problem_dir}/{encodec_dict['encoder_dir']}"
    if (
        "full_state" in cli_args["problem_dir"]
        and config_dict["encoder_decoder"]["decode"]
    ):
        decoder_cobasis_filename = f"{decoder_basis}_cobasis.npy"
        decoder_basis_filename = f"{decoder_basis}_basis.npy"
    else:
        decoder_cobasis_filename = None
        decoder_basis_filename = None

    if encodec_dict["save_embedded_data"]:
        config_dict["encoder_decoder"][
            "save_dir"
        ] = f"{problem_dir}/{encodec_dict['save_embedded_data']}"
    else:
        config_dict["encoder_decoder"]["save_dir"] = None
    config_dict["encoder_decoder"][
        "encoder_basis_filename"
    ] = f"{encoder_basis}_encoder_basis.npy"
    config_dict["encoder_decoder"][
        "encoder_cobasis_filename"
    ] = f"{encoder_basis}_encoder_cobasis.npy"
    config_dict["encoder_decoder"]["decoder_basis_filename"] = decoder_basis_filename
    config_dict["encoder_decoder"][
        "decoder_cobasis_filename"
    ] = decoder_cobasis_filename

    # config_dict['encoder_decoder']['reduced_data_filenames'] = ('X_reduced.npy','Y_reduced.npy','J_reduced.npy') #these files may not exist

    # Neural Network Architecture parameters #what if no encoder/decoder provided
    from numpy import load

    config_dict["nn"]["layer_width"] = (
        2
        * load(
            f"{config_dict['encoder_decoder']['encoder_decoder_dir']}{config_dict['encoder_decoder']['encoder_basis_filename']}"
        ).shape[1]
    )

    # Data (Directory Location/Training) parameters
    config_dict["data"][
        "samples_dir"
    ] = f"{problem_dir}/{config_dict['data']['samples_dir']}"
    if (
        config_dict["encoder_decoder"]["encode"]
        or config_dict["encoder_decoder"]["decode"]
    ):
        config_dict["data"]["data_filenames"] = (
            "X_data.npy",
            "fX_data.npy",
            "dfXdX_data.npy",
        )
    else:
        config_dict["data"]["data_filenames"] = (
            "X_data.npy",
            "fX_data.npy",
            "dfXdX_data.npy",
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
        "trained_nn_dir"
    ] = f"{problem_dir}/trained_nn/"  # make a CLI
    config_dict["network_serialization"][
        "loss_results_dir"
    ] = f"{problem_dir}/{config_dict['network_serialization']['loss_results_dir']}"  # make a CLI
    config_dict["network_serialization"][
        "nn_checkpoint_path"
    ] = "f{problem_dir}/trained_nn/"
    config_dict["network_serialization"]["nn"] = config_dict["nn"]
    random_seed = cli_args["run_seed"]
    # for key, cli_args[key] in name_string_keys:
    # += f"_{key}{cli_args[key]}"
    if cli_args["h1_weight"] != 0.0:
        config_dict["network_serialization"][
            "nn_save_name"
        ] += f"_h1weight{cli_args['h1_weight']}_seed{random_seed}"
    config_dict["network_serialization"][
        "nn_save_name"
    ] += f"_n_train{config_dict['data']['train_data_size']}_nepochs{config_dict['training']['n_epochs']}"
    config_dict["network_serialization"][
        "nn_save_name"
    ] += f"_depth{config_dict['nn']['depth']}_batchsize{config_dict['training']['batch_size']}"

    trained_approximator, training_results_dict = train_dino_in_embedding_space(
        random_seed=random_seed, embedded_training_config_dict=config_dict
    )

    #################################################################################
    # Save training metrics results to disk							                #
    #################################################################################
    # Disk I/O
    nn_serialize_dict = config_dict["network_serialization"]
    # logger = {'reduced':training_logger} #,'full': final_logger}
    nn_save_name = nn_serialize_dict["nn_save_name"]
    save_to_pickle(
        Path(
            f"{nn_serialize_dict['loss_results_dir']}/{nn_save_name}" ,
        ),
        training_results_dict,
    )

    #################################################################################
    # Save neural network parameters to disk (serialize the equinox pytrees)        #
    #################################################################################
    # Disk I/O
    if nn_serialize_dict["save_weights"]:
        # eqx nn weights serialization
        makedirs(nn_serialize_dict["trained_nn_dir"], exist_ok=True)
        eqx.tree_serialise_leaves(
            f"{nn_serialize_dict['trained_nn_dir']}{nn_save_name}.eqx",
            trained_approximator,
        )
        # eqx nn class serialization
        save_to_pickle(
            Path(nn_serialize_dict["trained_nn_dir"], nn_save_name),
            nn_serialize_dict["nn"],
        )
    #################################################################################
    # Save config file for reproducibility                                          #
    #################################################################################
    cli_dir = f"{problem_dir}/command_line_args"
    # Disk I/O
    save_to_pickle(Path(cli_dir, nn_save_name), config_dict) #save to json (human readable)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# for problem_dir in ...
#     for train_data_size in []:
#         python -m dinox -nn_save_name "april 13" -problem_dir "/storage/joshua/nonlinear_diffusion_reaction/problem/rel_noise_0.002_noise_stdev_0.0019357267998146984/" -train_data_size 2000
