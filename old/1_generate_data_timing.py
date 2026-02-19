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
    "-problem_name",
    "--problem_name",
    type=str,
    default="hyperelasticity",  # "nonlinear_diffusion_reaction", #"hyperelasticity",
    help="Problem name",
)

parser.add_argument(
    "-n_samples",
    "--n_samples",
    type=int,
    default=100,
    help="N prior/forward solve/jacobian samples",
)
parser.add_argument(
    "-n_iid_data_per_sample",
    "--n_iid_data_per_sample",
    type=int,
    default=10,
    help="N data samples per prior sample",
)

args = parser.parse_args()

import importlib

base_folder = args.save_base_folder
problem_name = args.problem_name
PDEproblem_dir = problem_name + "/"
model_module = importlib.import_module(problem_name + ".model")
settings = model_module.settings()  # random generation of points may hapepn here. Must only be done once.
################################################################################
# Set up the model

n_iid_data_per_sample = args.n_iid_data_per_sample
# I need an outer loop here on several noise stdevs:
# for i, rel_noise in enumerate(
#     settings["rel_noises"]
# ):  # (high variance may revert to the prior, low variance may revert too easily to near-gaussian around MAP ppoint)
#     # usually we use 0.01 = 1% of max = stdev, which correpsonds to very roughly 1% "SNR", to make the inference problem harder
#     # can try varying amounts
#     settings["rel_noise"] = rel_noise
#     settings["output_path"] = base_folder + PDEproblem_dir + "/figures/"
#     print(
#         f"Sampling for a noise model based on a relative max Linf noise of {100*rel_noise}%"
#     )
#     settings["seed"] = (
#         i  # different seed for each set of problems, what about hippylib seed?
#     )
#     pde, prior, misfit, mtrue, noise_stdev = model_module.model(settings)
#     BIPproblem_dir = f"{base_folder}{PDEproblem_dir}problem/rel_noise_{rel_noise}_noise_stdev_{noise_stdev}/"

#     model = hp.Model(pde, prior, misfit)

#     # Extract BiLaplacianPrior parameters as discrete matrices, so that we don't rely on hippylib prior instantiation afterwards
#     M_mat = dl.as_backend_type(prior.M).mat()
#     row, col, val = M_mat.getValuesCSR()
#     M_csc = ss.csr_matrix((val, col, row)).tocsc()
#     # from scipy.linalg import cholesky, eigh, eigh_tridiagonal, ldl

#     # L, d, P = ldl(M_csr)
#     # eigs, V = eigh_tridiagonal(np.diag(d), np.diag(d, 1))

#     # def M_sqrt_action(x):
#     #     return L @ (V * eigs) @ V.T @ L.T @ x
#     # from sksparse.cholmod import cholesky
#     # factor = cholesky(M_csr)
#     # L = factor.L().todense()
#     # print(factor.L())
#     # print(np.linalg.norm(L.dot(L.T) - M_csr.todense())) #/np.linalg.norm(M_csr.todense())

#     A_mat = dl.as_backend_type(prior.A).mat()
#     row, col, val = A_mat.getValuesCSR()
#     A_csr = ss.csr_matrix((val, col, row))
#     # sqrtMPetsc = dl.as_backend_type(prior.sqrtM).mat()
#     # row, col, val = (
#     #     sqrtMPetsc.getValuesCSR()
#     # )  # I think they only give the csr, so we convert
#     # sqrtMcsr = ss.csr_matrix((val, col, row))


#     # print(sqrtMcsr.nnz, sqrtMcsr.shape[0]*sqrtMcsr.shape[1], sqrtMcsr.shape[0],sqrtMcsr.shape[1], len(val))

#     prior_params_dir = BIPproblem_dir + "prior/"
#     # os.makedirs(prior_params_dir, exist_ok=True)
#     precision = A_csr.todense().T@np.linalg.solve(M_csc.todense(),A_csr.todense())
#     cov = np.linalg.inv(precision)
#     evals_precision, evecs_cov = np.linalg.eigh(precision)
#     cov_evals = 1/evals_precision
#     cov_sqrt = evecs_cov @ np.diag(np.sqrt(cov_evals))@ evecs_cov.T
#     # np.save("dense_cov_sqrt)

#     # np.save(prior_params_dir+ "square_dense_C_sqrt.npy", cov_sqrt)
#     # save_scipy_csr(
#     #     prior_params_dir + "M", M_csr
#     # )  # sort of redundant, each problems prior is the same, but whatever
#     # print(M_csr - load_jax_csr(data_dir + "prior/M").todense(). max())
#     # save_scipy_csr(
#     #     prior_params_dir + "sqrtM", sqrtMcsr
#     # )  # sort of redundant, each problems prior is the same, but whatever

#     # save_scipy_csr(
#     #     prior_params_dir + "A", A_csr
#     # )  # sort of redundant, each problems prior is the same, but whatever
#     # np.save(
#     #     prior_params_dir + "mean_function_coefficients.npy", prior.mean.get_local()
#     # )  # sort of redundant, each problems prior is the same, but whatever

#     #In the future, we should atually use batch sparse action from cupy, i.e. load sqrtMcsr to BCSR and
#     #apply batch sampling. Would also be idea once batch sparse solving is available from cuda
#     # np.save(prior_params_dir+ "C_sqrt.npy", (ss.linalg.spsolve(A_csr, sqrtMcsr)).T.todense() )
#     # np.save(prior_params_dir+ "sampling_dim.npy", sqrtMcsr.shape[1])


#     precision = A_csr.todense().T@np.linalg.solve(M_csc.todense(),A_csr.todense())
#     cov = np.linalg.inv(precision)
#     evals_precision, evecs_cov = np.linalg.eigh(precision)
#     cov_evals = 1/evals_precision
#     cov_sqrt = evecs_cov @ np.diag(np.sqrt(cov_evals))@ evecs_cov.T
# np.save(prior_params_dir+ "square_dense_C_sqrt.npy", cov_sqrt)

# Save data/observation parameters
# os.makedirs(BIPproblem_dir + "likelihood/", exist_ok=True)
# np.save(BIPproblem_dir + "likelihood/noise_stdev.npy", noise_stdev)
# np.save(
#     BIPproblem_dir + "likelihood/observation_targets_coordinates.npy",
#     settings["targets"],
# )

################################################################################
# Generate training data + synthetic data observations from likelihood
# print("Generating 50 problems out of which 4 will be chosen randomly")
# Vh = model.problem.Vh

# mesh = Vh[hp.STATE].mesh()

# B = model.misfit.B
# observable = hf.LinearStateObservable(model.problem, B)

# dQ = settings["dQ"]
# # Since the problems are
# output_basis = np.eye(dQ)

# dataGenerator = hf.DataGenerator(
#     observable, prior, misfit
# )  # just adding in misfit for access to the problems noise

# nsamples = args.n_samples
# # os.makedirs(f"{base_folder}{problem_dir}",exist_ok=True)

# true_data_dir = f"{BIPproblem_dir}true_samples/"

# dataGenerator.generate(
#     50,
#     derivatives=(1, 0),
#     output_basis=output_basis,
#     n_data_per_sample=n_iid_data_per_sample,
#     data_dir=true_data_dir,
# )
# This is the data used by DINOX to train the dino
# mqy_data = np.load(true_data_dir + "mqy_data.npz")
# np.save(true_data_dir + "X_data.npy", mqy_data["m_data"])
# np.save(true_data_dir + "fX_data.npy", mqy_data["q_data"])
# np.save(
#     true_data_dir + "dfXdX_data.npy",
#     np.load(true_data_dir + "JstarPhi_data.npz")["JstarPhi_data"],
# )
# # This will be used if we need bayesian inference problems (pick an X and Y, for hte true M and true noisy data)
# np.save(true_data_dir + "Y_data.npy", mqy_data["y_data"])


for i, BIPproblem_sub_dir in enumerate(
    os.listdir(f"{base_folder}{PDEproblem_dir}problem")
):  # (high variance may revert to the prior, low variance may revert too easily to near-gaussian around MAP ppoint)
    # usually we use 0.01 = 1% of max = stdev, which correpsonds to very roughly 1% "SNR", to make the inference problem harder
    # can try varying amounts
    # settings["rel_noise"] = rel_noise
    settings["output_path"] = None
    settings["seed"] = i  # different seed for each set of problems, what about hippylib seed?
    print("loading BIP dir", BIPproblem_sub_dir)

    pde, prior, misfit, mtrue, _ = model_module.model(settings)
    BIPproblem_dir = f"{base_folder}{PDEproblem_dir}problem/" + BIPproblem_sub_dir + "/"
    noise_stdev = np.load(BIPproblem_dir + "likelihood/noise_stdev.npy")
    misfit.noise_variance = noise_stdev * noise_stdev

    model = hp.Model(pde, prior, misfit)

    ################################################################################
    # Generate training data + synthetic data observations from likelihood

    Vh = model.problem.Vh

    mesh = Vh[hp.STATE].mesh()

    B = model.misfit.B
    observable = hf.LinearStateObservable(model.problem, B)

    dQ = settings["dQ"]
    # Since the problems are
    output_basis = np.eye(dQ)

    dataGenerator = hf.DataGenerator(
        observable, prior, misfit
    )  # just adding in misfit for access to the problems noise

    nsamples = args.n_samples

    data_dir = f"{BIPproblem_dir}samples_temp/"

    # saves m, f(m), y = f(m) + eta_i for several eta_i (n_independent_data_per_sample #)
    dataGenerator.generate(
        nsamples,
        derivatives=(1, 0),
        output_basis=output_basis,
        n_data_per_sample=n_iid_data_per_sample,
        data_dir=data_dir,
        compress=False,
    )

    # resave as individual uncompressed arrays, for use with dinox

    # This is the data used by DINOX to train the dino
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
