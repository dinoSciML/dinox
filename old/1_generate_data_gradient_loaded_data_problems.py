import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import math
import sys

import dolfin as dl
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ss

sys.path.append(os.environ.get("HIPPYLIB_PATH"))
import hippylib as hp

sys.path.append(os.environ.get("HIPPYFLOW_PATH"))
import hippyflow as hf


def save_scipy_csr(filename, csr_matrix):  # place this where?
    filename += "_csr"
    np.save(filename + "_data", csr_matrix.data)
    np.save(filename + "_indices", csr_matrix.indices)
    np.save(filename + "_indptr", csr_matrix.indptr)
    np.save(filename + "_shape", np.array(list(csr_matrix.shape)))


################################################################################
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-save_base_folder",
    "--save_base_folder",
    type=str,
    default="/storage/joshua/",
    help="Folder to save everything in",
)

parser.add_argument(
    "-rel_noise",
    "--rel_noise",
    type=str,
    default="rel_noise_0.002_noise_stdev_0.0019357267998146984",  # "rel_noise_0.002_noise_stdev_0.002860335630134797",
    help="relative_noise",
)
parser.add_argument(
    "-problem_name",
    "--problem_name",
    type=str,
    default="nonlinear_diffusion_reaction",  # "nonlinear_diffusion_reaction", #"hyperelasticity",
    help="Problem name",
)
parser.add_argument(
    "-n_problems",
    "-n_problems",
    type=int,
    default=15,
    help="Number of problems to generate data for (random data)",
)

parser.add_argument(
    "-n_samples",
    "--n_samples",
    type=int,
    default=2100,  # 1000
    help="N prior/forward solve/jacobian samples",
)
parser.add_argument(
    "-n_iid_data_per_sample",
    "--n_iid_data_per_sample",
    type=int,
    default=10,
    help="N data samples per prior sample",
)
parser.add_argument(
    "-process",
    "--process",
    type=int,
    default=1,
    help="Process number, e.g. 0 to 49",
)


args = parser.parse_args()

import importlib

base_folder = args.save_base_folder
problem_name = args.problem_name
PDEproblem_dir = problem_name + "/"
model_module = importlib.import_module(problem_name + ".model")
settings = model_module.settings()  # random generation of points may hapepn here. Must only be done once.
n_problems = args.n_problems
sample_start_idx = args.process * args.n_samples
################################################################################
# Set up the model

n_iid_data_per_sample = args.n_iid_data_per_sample
# I need an outer loop here on several noise stdevs:
# for i, rel_noise in enumerate(
#     settings["rel_noises"]
# ):  # (high variance may revert to the prior, low variance may revert too easily to near-gaussian around MAP ppoint)
# usually we use 0.01 = 1% of max = stdev, which correpsonds to very roughly 1% "SNR", to make the inference problem harder
# can try varying amounts

rel_noise_folder = args.rel_noise
left, right = rel_noise_folder.rsplit("_", 1)
rel_noise = float(right)  # float(rel_noise_folder.rsplit('_', 1)[0])
# settings["rel_noise"] = rel_noise
print(f"Sampling for a noise model based on a relative max Linf noise of {100*rel_noise}%")
settings["seed"] = 0  # what about hippylib seed?
pde, prior, misfit, mtrue, noise_stdev = model_module.model(settings)
noise_stdev = rel_noise
BIPproblem_dir = f"{base_folder}{PDEproblem_dir}amortizedProblem/"  # rel_noise_{rel_noise}_noise_stdev_{noise_stdev}/"

model = hp.Model(pde, prior, misfit)

# Extract BiLaplacianPrior parameters as discrete matrices, so that we don't rely on hippylib prior instantiation afterwards
M_mat = dl.as_backend_type(prior.M).mat()
row, col, val = M_mat.getValuesCSR()
M_csc = ss.csr_matrix((val, col, row)).tocsc()

A_mat = dl.as_backend_type(prior.A).mat()
row, col, val = A_mat.getValuesCSR()
A_csr = ss.csr_matrix((val, col, row))

prior_params_dir = BIPproblem_dir + "prior/"
os.makedirs(prior_params_dir, exist_ok=True)
precision = A_csr.todense().T @ np.linalg.solve(M_csc.todense(), A_csr.todense())
cov = np.linalg.inv(precision)
evals_precision, evecs_cov = np.linalg.eigh(precision)
cov_evals = 1 / evals_precision
cov_sqrt = evecs_cov @ np.diag(np.sqrt(cov_evals)) @ evecs_cov.T
# np.save("dense_cov_sqrt)

np.save(prior_params_dir + "square_dense_C_scov_evalsqrt.npy", cov_sqrt)
del cov_sqrt, evals_precision, evecs_cov, cov_evals, M_csc, A_csr

# Save data/observation parameters
os.makedirs(BIPproblem_dir + "likelihood/", exist_ok=True)
np.save(BIPproblem_dir + "likelihood/noise_stdev.npy", noise_stdev)
np.save(
    BIPproblem_dir + "likelihood/observation_targets_coordinates.npy",
    settings["targets"],
)

################################################################################
# Generate training data + synthetic data observations from likelihood
print("Generating 100 problems out of which 10 will be chosen randomly")
Vh = model.problem.Vh
mesh = Vh[hp.STATE].mesh()
B = model.misfit.B


observable = hf.LinearStateObservable(model.problem, B)

dQ = settings["dQ"]
# Since the problems are
output_basis = np.eye(dQ)

misfit.noise_variance = noise_stdev * noise_stdev
misfit.n_problems = n_problems
dataGenerator = hf.DataGenerator(observable, prior, misfit)  # just adding in misfit for access to the problems noise

nsamples = args.n_samples
# os.makedirs(f"{base_folder}{problem_dir}",exist_ok=True)

true_data_dir = f"{BIPproblem_dir}true_samples/"

# Generate synthetic data observations from likelihood
dataGenerator.generate(
    15,
    0,
    derivatives=(0, 0, 0),
    output_basis=output_basis,
    n_data_per_sample=n_iid_data_per_sample,
    data_dir=true_data_dir,
    save=(0 == sample_start_idx),
)

# This is the data used by DINOX to train the dino
if sample_start_idx == 0:
    mqy_data = np.load(true_data_dir + "mqy_data.npz")
    np.save(true_data_dir + "X_data.npy", mqy_data["m_data"])
    np.save(true_data_dir + "fX_data.npy", mqy_data["q_data"])
    # This will be used if we need bayesian inference problems (pick an X and Y, for hte true M and true noisy data)
    np.save(true_data_dir + "Y_data.npy", mqy_data["y_data"])

    # pick 0'th data specifically for inference
    for i in range(15):
        os.makedirs(f"{BIPproblem_dir}BIP_data/", exist_ok=True)
        np.save(f"{BIPproblem_dir}BIP_data/true_Y_{i}.npy", mqy_data["y_data"][i][0])
        np.save(f"{BIPproblem_dir}BIP_data/true_X_{i}.npy", mqy_data["m_data"][i])
else:
    mqy_data = dict()
    while True:
        try:
            mqy_data["y_data"] = np.load(true_data_dir + "Y_data.npy")
            break
        except:
            pass
# choose the 15 BIPs and fix them!


settings["rel_noise"] = rel_noise
settings["output_path"] = None
settings["seed"] = 1  # different seed for each set of problems, what about hippylib seed?

nsamples = args.n_samples
data_dir = f"{BIPproblem_dir}samples/"
# # saves m, f(m), y = f(m) + eta_i for several eta_i (n_independent_data_per_sample #)

print(f"Now generating {nsamples} samples, starting from index {sample_start_idx}")
# run 85 instances of 1000 samples
dataGenerator.generate(  # how do we add in f"{BIPproblem_dir}BIP_data/true_Y_{i}.npy" into the results???
    nsamples,
    sample_start_idx,
    derivatives=(1, 0, 1),
    y_samples=mqy_data["y_data"][:, 0],
    output_basis=output_basis,
    n_data_per_sample=n_iid_data_per_sample,
    data_dir=data_dir,
    compress=False,
    clean_up=False,
)


# # resave as individual uncompressed arrays, for use with dinox

# # This is the data used by DINOX to train the dino
# mqy_data = np.load(data_dir + "mqy_data.npz")
# np.save(data_dir + "X_data.npy", mqy_data["m_data"])
# np.save(data_dir + "fX_data.npy", mqy_data["q_data"])
# np.save(
#     data_dir + "dfXdX_data.npy",
#     np.load(data_dir + "JstarPhi_data.npz")["JstarPhi_data"],
# )
# np.save(
#     data_dir + "Y_data.npy", mqy_data["y_data"]
# )  # in case we need them for the future


# save cli_args save_base_folder, problem_name, n_samples, n_iid_data_per_sample
