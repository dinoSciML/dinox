import math
import os
import sys

import dolfin as dl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ss

sys.path.append(os.environ.get("HIPPYLIB_PATH"))
import hippylib as hp

sys.path.append(os.environ.get("HIPPYFLOW_PATH"))
################################################################################
import argparse

import hippyflow as hf
import scipy.spatial as spspa
from sklearn.cluster import SpectralClustering

parser = argparse.ArgumentParser()

parser.add_argument(
    "-save_base_folder",
    "--save_base_folder",
    type=str,
    default="/storage/joshua/",
    help="Folder to save everything in",
)

parser.add_argument(
    "-problem_name",
    "--problem_name",
    type=str,
    default="hyperelasticity",
    help="Problem name",
)


args = parser.parse_args()

import importlib

base_folder = args.save_base_folder
problem_name = args.problem_name
PDEproblem_dir = problem_name + "/"
model_module = importlib.import_module(problem_name + ".model")
settings = (
    model_module.settings()
)  # random generation of points may hapepn here. Must only be done once.
################################################################################
# Set up the model

# n_iid_data_per_sample = args.n_iid_data_per_sample
# I need an outer loop here on several noise stdevs:
for i, BIPproblem_sub_dir in enumerate(
    os.listdir(f"{base_folder}{PDEproblem_dir}problem")
):  # (high variance may revert to the prior, low variance may revert too easily to near-gaussian around MAP ppoint)
    # usually we use 0.01 = 1% of max = stdev, which correpsonds to very roughly 1% "SNR", to make the inference problem harder
    # can try varying amounts
    # settings["output_path"] = base_folder+PDEproblem_dir+"/figures/"
    # settings["seed"] = i #different seed for each set of problems, what about hippylib seed?
    BIPproblem_dir = f"{base_folder}{PDEproblem_dir}problem/{BIPproblem_sub_dir}/"

    # noise_stdev = np.load(BIPproblem_dir+"likelihood/noise_stdev.npy")
    # targets = np.load(BIPproblem_dir+"likelihood/observation_targets_coordinates.npy")
    # settings['targets'] = targets
    # misfit.noise_variance = noise_stdev*noise_stdev

    # pde, prior, misfit, mtrue, _ = model_module.model(settings)

    # model = hp.Model(pde, prior, misfit)

    # ################################################################################
    # # Generate training data + synthetic data observations from likelihood

    # Vh = model.problem.Vh

    # mesh = Vh[hp.STATE].mesh()

    # B = model.misfit.B
    # observable = hf.LinearStateObservable(model.problem,B)

    # dQ = settings['dQ']
    # # Since the problems are
    # output_basis = np.eye(dQ)

    true_data_dir = f"{BIPproblem_dir}true_samples/"

    # This is the data used by DINOX to train the dino
    # mqy_data = np.load(true_data_dir + "mqy_data.npz")
    # np.load(true_data_dir + "X_data.npy", mqy_data["m_data"])
    X_data = np.load(true_data_dir + "X_data.npy")

    Y_data = np.load(true_data_dir + "Y_data.npy")
    print(Y_data.shape)
    clustering = SpectralClustering(
        n_clusters=100, assign_labels="discretize", random_state=0
    ).fit(Y_data[:, 0])
    clustering.labels_

    samples_Y = []
    samples_X = []

    sample_indices = []
    for i in range(4):
        sample_idx = np.where(clustering.labels_ == i)[0][0]
        sample_indices.append(sample_idx)
        samples_Y.append(Y_data[sample_idx, 0])
        samples_X.append(X_data[sample_idx])
    os.makedirs(f"{BIPproblem_dir}BIP_data/", exist_ok=True)

    np.save(
        f"{BIPproblem_dir}BIP_data/true_sample_indices.npy", np.array(sample_indices)
    )
    for i in range(4):
        np.save(f"{BIPproblem_dir}BIP_data/true_Y_{i}.npy", samples_Y[i])
        np.save(f"{BIPproblem_dir}BIP_data/true_X_{i}.npy", samples_X[i])

    # Save prior parametersls


# save cli_args save_base_folder, problem_name, n_samples, n_iid_data_per_sample

# def choose_4_random_data_and_save_to_disk(problems_base_dir, problem_name)

#     problem_dir = problem_name+"/"
#     model_module = importlib.import_module(f"{problems_base_dir}{problem_name}+".model")
#     settings = model_module.settings()
#     ################################################################################
#     # Set up the model

#     n_iid_data_per_sample =

#     settings["rel_noise"] = rel_noise

#     pde, prior, misfit, mtrue, noise_stdev = model_module.model(settings)


#     data = []
#     while True:
#         pick random data in y_data[24000:]
#         #AS LONG AS THE L2 DISTANCE > PERCENTAGE OF L2 NORM
#         data.append()

#     save BIP data indices to disk


# def instantiate_hippylib_BIP_model(problem_...., BIP_data_index_file)

#     pde, prior, misfit, mtrue, noise_stdev = model_module.model(settings)
#     misfit.d = data[index[i]]
#     return hp.Model(pde, prior, misfit)
