# THIS and all data generation/model problem codes should go in a common "problems" subrepo of DinoSciML i think. and people can refactor their codes
# once we come to a consensus of the easiest format for problems.
import os
import sys
import time

import dolfin as dl
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.environ.get("HIPPYLIB_PATH"))
import hippylib as hp

sys.path.append(os.environ.get("HIPPYFLOW_PATH"))
################################################################################
import argparse

import hippyflow as hf

parser = argparse.ArgumentParser()

parser.add_argument(
    "-save_base_folder",
    "--save_base_folder",
    type=str,
    default="/storage/joshua/",
    help="Folder to save everything in",
)

parser.add_argument("-basis_type", "--basis_type", type=str, default="as", help="as or kle")
parser.add_argument("-rank", "--rank", type=int, default=200, help="Active subspace rank")
parser.add_argument(
    "-oversample",
    "--oversample",
    type=int,
    default=10,
    help="Active subspace oversample",
)
parser.add_argument(
    "-problem_name",
    "--problem_name",
    type=str,
    default="nonlinear_diffusion_reaction",
    help="Problem name",
)
parser.add_argument(
    "-ndata",
    "--ndata",
    type=int,
    default=1000,
    help="Number of samples to use for active subspace (if -base_type ==as)",
)

args = parser.parse_args()

################################################################################
# Parameters

rank = args.rank
oversample = args.oversample

################################################################################
# Set up the model
import importlib

base_folder = args.save_base_folder
problem_name = args.problem_name
PDEproblem_dir = problem_name + "/"
model_module = importlib.import_module(problem_name + ".model")
settings = model_module.settings()  # only uses the prior settings
_, prior, misfit, _, _ = model_module.model(settings)  # only uses the prior settings

assert dl.MPI.comm_world.size == 1, print("Not thought out in other cases yet")

for i, BIPproblem_sub_dir in enumerate(
    os.listdir(f"{base_folder}{PDEproblem_dir}problem")
):  # (high variance may revert to the prior, low variance may revert too easily to near-gaussian around MAP ppoint)
    print(f"Computing bases for {BIPproblem_sub_dir}")
    BIPproblem_dir = f"{base_folder}{PDEproblem_dir}problem/" + BIPproblem_sub_dir + "/"
    data_dir = BIPproblem_dir + "samples/"
    encoder_save_dir = BIPproblem_dir + "encoder/"
    os.makedirs(encoder_save_dir, exist_ok=True)
    if args.basis_type.lower() == "as":
        ################################################################################
        # Load the data

        all_data = np.load(data_dir + "mq_data.npz")
        JTPhi_data = np.load(data_dir + "JstarPhi_data.npz")
        noise_stdev = np.load(BIPproblem_dir + "likelihood/noise_stdev.npy")
        m_data = all_data["m_data"][: args.ndata]
        q_data = all_data["q_data"][: args.ndata]
        PhiTJ_data = np.transpose(JTPhi_data["JstarPhi_data"], (0, 2, 1))[: args.ndata] / (noise_stdev**2.0)

        print("m_data.shape = ", m_data.shape)
        print("q_data.shape = ", q_data.shape)
        print("PhiTJ_data.shape = ", PhiTJ_data.shape)

        ################################################################################
        # Instance JTJ operator
        print("Loading JTJ")
        JTJ_operator = hf.MeanJTJfromDataOperator(PhiTJ_data, prior)
        # Set up the Gaussian random
        m_vector = dl.Vector()
        JTJ_operator.init_vector_lambda(m_vector, 0)
        Omega = hp.MultiVector(m_vector, rank + oversample)
        hp.parRandom.normal(1.0, Omega)

        t0 = time.time()
        print("Beginning doublePassG")
        if hasattr(prior, "R"):
            d_GN, V_GN = hp.doublePassG(JTJ_operator, prior.R, prior.Rsolver, Omega, rank, s=1)
        else:
            d_GN, V_GN = hp.doublePassG(JTJ_operator, prior.Hlr, prior.Hlr, Omega, rank, s=1)

        print("doublePassG took ", time.time() - t0, "s")

        encoder_basis = hf.mv_to_dense(V_GN)

        # Compute the projector RV_r from the basis
        RV_GN = hp.MultiVector(V_GN[0], V_GN.nvec())
        RV_GN.zero()
        hp.MatMvMult(prior.R, V_GN, RV_GN)
        # C^-1 encoder

        encoder_cobasis = hf.mv_to_dense(RV_GN)
        # need C^1/2 and C^-1/2 == Rsqrtinv and Rsqrt
        # cna juse A M Ainverse or

        # plot bases

        check_orth = True
        if check_orth:
            PsistarPsi = encoder_cobasis.T @ encoder_basis
            orth_error = np.linalg.norm(PsistarPsi - np.eye(PsistarPsi.shape[0]))
            print("||Psi^*Psi - I|| = ", orth_error)
            assert orth_error < 1e-5
        print(encoder_basis.shape, encoder_cobasis.shape)
        np.save(encoder_save_dir + "AS_encoder_basis", encoder_basis)
        # d_GN = d_GN
        np.save(encoder_save_dir + "AS_d_GN", d_GN)
        np.save(
            encoder_save_dir + "AS_encoder_cobasis",
        )

        fig, ax = plt.subplots()
        ax.semilogy(np.arange(len(d_GN)), d_GN)
        print()

        ax.set(
            xlabel="index",
            ylabel="eigenvalue",
            title="GEVP JTJ/sigma^2 spectrum",
        )  # TODO offline plot these! someone else do this..
        ax.grid()

        fig.savefig("JTJsigma^-2_eigenvalues.pdf")

        # print("Loading JJT")
        # JJT_operator = hf.MeanJJTfromDataOperator(PhiTJ_data, prior)
        # # Set up the Gaussian random
        # m_vector = dl.Vector()
        # JTJ_operator.init_vector_lambda(m_vector, 0)
        # Omega = hp.MultiVector(m_vector, rank + oversample)
        # hp.parRandom.normal(1.0, Omega)

        # t0 = time.time()
        # print("Beginning doublePassG")
        # if hasattr(prior, "R"):
        #     d_GN, V_GN = hp.doublePassG(
        #         JTJ_operator, prior.R, prior.Rsolver, Omega, rank, s=1
        #     )
        # else:
        #     d_GN, V_GN = hp.doublePassG(
        #         JTJ_operator, prior.Hlr, prior.Hlr, Omega, rank, s=1
        #     )

    elif args.basis_type.lower() == "kle":
        KLE = hf.KLEProjector(prior)
        KLE.parameters["rank"] = rank
        KLE.parameters["oversampling"] = oversample
        KLE.parameters["save_and_plot"] = False

        d_KLE, kle_basis, kle_projector = KLE.construct_input_subspace()

        encoder_basis = hf.mv_to_dense(kle_basis)
        encoder_cobasis = hf.mv_to_dense(kle_projector)

        check_orth = True
        if check_orth:
            PsistarPsi = encoder_cobasis.T @ encoder_basis
            orth_error = np.linalg.norm(PsistarPsi - np.eye(PsistarPsi.shape[0]))
            print("||Psi^*Psi - I|| = ", orth_error)
            assert orth_error < 1e-5

        np.save(encoder_save_dir + "KLE_basis", encoder_basis)
        np.save(encoder_save_dir + "KLE_d", d_KLE)
        np.save(encoder_save_dir + "KLE_cobasis", encoder_cobasis)

    else:
        raise
