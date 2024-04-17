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
from numpy import load
from pathlib import Path
from typing import Any, Dict
import jax

from .data_utilities import save_to_pickle, sub_dict
from .train import train_nn_in_embedding_space, save_training_results

jax.config.update("jax_enable_x64", True)

def define_cli_arguments(cli: ArgumentParser) -> None:
    """
    Defines and adds command-line arguments for configuring the neural network training process.

    Parameters
    ----------
    cli : ArgumentParser
        The command-line argument parser instance.
    """
    # Required CLI arguments
    cli.add_argument("-problemDir", type=str, required=True, help="Directory where problem data is located.")
    cli.add_argument("-nTrain", type=int, required=True, help="# samples to use as the training dataset.")
    cli.add_argument("-nnSavePrefix", type=str, required=True, help="Filename Prefix for saving the trained neural network.")

    # Neural Network architecture parameters
    cli.add_argument("-architecture", type=str, default="genericDense",  choices=['genericDense','rnn'], help="Type of NN architecture.")
    cli.add_argument("-activation", type=str, default="gelu", help="NN Activation function.")
    cli.add_argument("-depth", type=int, default=8, help="Number of layers in the NN.")

    # Data handling parameters
    cli.add_argument("-nTest", type=int, default=5000, help="# samples to use as the testing dataset.")

    # Optimization parameters
    cli.add_argument("-loss", type=str, default='h1', choices=['h1','l2'], help="H1 or L2 loss?")
    cli.add_argument("-batchSize", type=int, default=25, help="# samples per batch during training.")
    cli.add_argument("-optaxOptimizer", type=str, default="adam", help="Optax optimizer to trian with")
    cli.add_argument("-nEpochs", type=int, default=500, help="# epochs for training.")
    cli.add_argument("-stepSize", type=float, default=1e-3, help="Learning rate for optax optimizer.")

    # Encoder/Decoder parameters
    cli.add_argument("-encoder", type=str, default="as",choices=['as','kle', ''], help="Type of encoder used.")
    cli.add_argument("-decoder", type=str, default="", choices=['pod', 'jjt',''], help="Type of decoder used.")
    cli.add_argument("-saveEmbeddedData", action="store_true", help="Flag to save embedded training data.")

    # Other 
    cli.add_argument("-runSeed", type=int, default=777, help="Seed for random # generation.")

    # # cli.add_argument(
    # #     "-fixed_input_rank",
    # #     dest="fixed_input_rank",
    # #     required=False,
    # #     default=200,
    # #     help="rank for input of AS network",
    # #     type=int,
    # # )
    # # cli.add_argument(
    # #     "-fixed_output_rank",
    # #     dest="fixed_output_rank",
    # #     required=False,
    # #     default=50,
    # #     help="rank for output of AS network",
    # #     type=int,
    # # )
    # # cli.add_argument( #reintroduce if we need the dimension of the problem to be lower than the number of bases
    # # given by AS
    # #     "-truncation_dimension",
    # #     dest="truncation_dimension",
    # #     required=False,
    # #     default=200,
    # #     help="truncation dimension for low rank networks",
    # #     type=int,
    # # )

def create_config_dict(cli_args: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Constructs a configuration dictionary for neural network training from command-line
    arguments, organizing the settings into categories such as network architecture,
    data management, training parameters, and serialization settings.

    Parameters
    ----------
    cli_args : Dict[str, Any]
        Dictionary containing all command-line arguments.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        A structured configuration dictionary with nested categories for different
        aspects of the setup and training process.

    Example
    -------
    >>> cli_args = {
        'runSeed': 42,
        'problemDir': '/path/to/data',
        'nnSavePrefix': 'April17',
        'architecture': 'generic_dense',
        'activation': 'gelu',
        'depth': 8,
        'nTrain': 2000,
        'nTest': 5000,
        'batchSize': 25,
        'optaxOptimizer': 'adam',
        'nEpochs': 500,
        'stepSize': 1e-3,
        'loss': 'l2',
        'encoder': 'as',
        'decoder': 'pod',
        'saveEmbeddedData': True,
    }
    >>> config = create_config_dict(cli_args)
    >>> print(config['nn']['architecture'])
    'generic_dense'
    """
    nn_keys = ("architecture", "depth", "activation") #layer_width defined later
    data_keys = ("nTrain", "nTest")
    training_keys = ("stepSize", "batchSize", "optaxOptimizer", "nEpochs", "loss")
    results_keys = ("nnSavePrefix", )
    encoder_decoder_keys = (
        "encoder",
        "decoder",
        "saveEmbeddedData",
    )
    # This line has to be called in the same scope as the keys defined above
    config = {
        k.removesuffix("_keys"): sub_dict(super_dict=cli_args, keys=v)
        for k, v in locals().items()
        if k.endswith("_keys")
    }

    #Create full NN name (based on parameters)
    nn_save_name = cli_args["nnSavePrefix"]
    name_string_keys = ['architecture','depth','loss','runSeed','nEpochs','batchSize','nTrain']
    for key in name_string_keys:
        nn_save_name += f"_{key}{cli_args[key]}"
    print("nn_save_name ",nn_save_name)


    # Additional configuration related to file paths and directories
    problem_dir = cli_args["problemDir"]

    #Load/Save paths
    config["data"]["dir"] = Path(problem_dir, "samples")
    config["data"]["N"] = load(Path(config["data"]["dir"],"X_data.npy")).shape[0]
    config["encoder_decoder"]["dir"] = Path(problem_dir, "encoder"),
    config["encoder_decoder"]["save_dir"] = Path(problem_dir, "samples") if cli_args["saveEmbeddedData"] else None
    config["data"]["filenames"] = ("X_data.npy", "fX_data.npy", "dfXdX_data.npy") #User should be able to specify this...

    config["results"] = {}
    config["results"]["nn_config"] = config["nn"]
    config["results"]["nn_class_path"] = Path(problem_dir, "trained_nn", nn_save_name, ".eqx_class")
    config["results"]["nn_weights_path"] = Path(problem_dir, "trained_nn", nn_save_name, ".eqx")
    config["results"]["training_metrics_path"] = Path(problem_dir, "training_metrics", nn_save_name, ".pkl")
    config["config_path"] = Path(problem_dir, "config", nn_save_name, ".pkl")

    encodec_config = config["encoder_decoder"]
    if encodec_config["encoder"]:
        config["encoder_decoder"][
            "encoder_basis_path"
        ] = Path(problem_dir, "encoder", f"{encodec_config['encoder'].upper()}_encoder_basis.npy")
        config["encoder_decoder"][
            "encoder_cobasis_path"
        ] = Path(problem_dir, "encoder", f"{encodec_config['encoder'].upper()}_encoder_cobasis.npy")
    if encodec_config["decoder"]:
        config["encoder_decoder"][
            "decoder_basis_path"
        ] = Path(problem_dir, "decoder", f"{encodec_config['decoder'].upper()}_decoder_basis.npy")
        config["encoder_decoder"][
            "decoder_cobasis_path"
        ] = Path(problem_dir, "decoder", f"{encodec_config['decoder'].upper()}_decoder_cobasis.npy")
        # Configure paths for encoder and decoder if used
        
        # if (
        #     "full_state" in cli_args["problem_dir"]
        #     and config["encoder_decoder"]["decode"]
        # ):
        #     decoder_cobasis_filename = f"{encodec_dict["decoder"].upper()}_cobasis.npy"
        #     decoder_basis_filename = f"{encodec_dict["decoder"].upper()}_basis.npy"
        # else:
        #     decoder_cobasis_filename = None
        #     decoder_basis_filename = None

    # Neural Network Architecture parameters #what if no encoder/decoder provided
    config["nn"]["layer_width"] = \
        2 * load(encodec_config['encoder_basis_path']).shape[1]
    
    return config

# TODO: add decorator for saving the returns of functions... rather than dealing with
# TODO: switch from argparse to click, use @click.option()
# TODO: use a logger for print statementes ('verbose=True') rather than printing
def main() -> int:
    """
    Main function to setup and run the training of a neural network using DINO
    (Derivative-Informed Neural Operator) architecture. It handles parsing command-line
    arguments, setting up data and network configurations, initiating training, and
    saving the results and model configurations.

    Returns
    -------
    int
        Exit status code: 0 indicates successful execution, other values indicate errors.

    Notes
    -----
    This function defines and processes command-line arguments to configure various
    aspects of neural network training, including architecture settings, data handling,
    optimization parameters, and paths for saving outputs. It leverages external libraries
    such as JAX, CuPy, and NumPy for efficient computation and data management. The function
    ensures all configurations are logged and saved for reproducibility. It encapsulates
    the workflow in a series of discrete steps that include:
    
    1. Parsing command-line arguments.
    2. Configuring neural network and data settings.
    3. Training the neural network.
    4. Saving training results and network parameters to disk.

    Examples
    --------
    Typical usage often involves invoking this script from the command line. An example
    command would be:
    python -m dinox -runSeed 42 -problemDir '/path/to/data' -nnSavePrefix 'Feb23' 
    
    Raises
    ------
    AssertionError
        If any configurations or paths are incorrect, potentially during the checks for
        data sizes, directory existence, or file operations.
    """
    cli = ArgumentParser(add_help=True)
    define_cli_arguments(cli)
    cli_args = vars(cli.parse_args())


    ###################################################################################
    # Define the keys for each configuration dict (*_keys is a required naming
    # convention here, since we define the configuration dict names by *
    ##################################################################################

    # Create a heirarchical configuration dictionary from CLI arguments
    config = create_config_dict(cli_args)
    # Save configuration for reproducibility
    save_to_pickle(config['config_path'], config)  

    # ESS, max weight,  k-fold cross validation (can do this for training lazymaps)

    # Perform the training of the neural network
    trained_approximator, results = train_nn_in_embedding_space(
        random_seed=cli_args['runSeed'], embedded_training_config=config
    )

    save_training_results(results=results, nn=trained_approximator, config=config['results'])
    return 0


if __name__ == "__main__":
    sys.exit(main())


