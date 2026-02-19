# Author: Lianghao Cao
# 09/24/2023

import argparse
import math
import os
import sys

import numpy as np

sys.path.append(os.environ.get("HIPPYLIB_PATH"))
from helmholtz_model import helmholtz_model
from helmholtz_settings import helmholtz_settings

from ...lazydinos.modeling.hippylib_modified import dl, hp

# sys.path.insert(0, '../')

sys.path.append("../../")
import lazydinos as ld
import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=20)
import matplotlib

matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath, mathrsfs}"

STATE, PARAMETER, ADJOINT = 0, 1, 2
dl.set_log_level(dl.LogLevel.ERROR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dr inference")

    parser.add_argument("--process_id", default=0, type=int, help="The processor ID")

    args = parser.parse_args()

    process_id = args.process_id

    settings = helmholtz_settings()

    np.random.seed(seed=settings["seed"])

    output_path = settings["output_path"]
    model = helmholtz_model(settings)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    save_path = output_path + "/chain_" + str(process_id) + "/"
    os.makedirs(save_path, exist_ok=True)

    settings["save_path"] = save_path
    settings["process_id"] = process_id

    mcmc = ld.gpCN(model)
    mcmc.parameters["process_id"] = settings["process_id"]  # ID for multiprocess run
    mcmc.parameters["save_path"] = settings["save_path"]  # Output path for saving parameters and cost evolution
    mcmc.parameters["number_of_samples"] = settings["number_of_samples"]  # The total number of samples
    mcmc.parameters["output_frequency"] = settings[
        "output_frequency"
    ]  # The number of time to print to screen or plot data misfit evolution
    if settings["verbose"]:
        mcmc.parameters["print_level"] = 1  # Negative to not print
    else:
        mcmc.parameters["print_level"] = -1
    mcmc.parameters["step_size"] = settings["step_size"]  # The step size for MALA proposal
    mcmc.parameters["burn_in"] = settings["burn_in"]  # The number of burn-in samples
    mcmc.run()
