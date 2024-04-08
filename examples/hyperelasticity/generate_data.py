import math
import numpy as np
import scipy.sparse as ss
import matplotlib.pyplot as plt
import dolfin as dl
import sys, os
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp
sys.path.append(os.environ.get('HIPPYFLOW_PATH'))
import hippyflow as hf
# from dinox.data_utilities import load_1D_jax_array_direct_to_gpu
import jax.experimental.sparse as jsparse

def save_scipy_csr(filename, csr_matrix): #place this where?
    filename+="_csr"
    np.save(filename+"_data", csr_matrix.data)
    np.save(filename+"_indices", csr_matrix.indices)
    np.save(filename+"_indptr", csr_matrix.indptr)
    np.save(filename+"_shape", np.array(list(csr_matrix.shape)))
# def load_jax_csr(filename):
#     filename+="_csr"
#     data = load_1D_jax_array_direct_to_gpu(filename+"_data.npy")
#     indices = load_1D_jax_array_direct_to_gpu(filename+"_indices.npy")
#     ind_ptr = load_1D_jax_array_direct_to_gpu(filename+"_ind_ptr.npy")
#     shape = load_1D_jax_array_direct_to_gpu(filename+"_shape.npy")
    # return jsparse.CSR(data, indices, ind_ptr, shape=tuple(shape))

from hyperelasticity_model import hyperelasticity_model, hyperelasticity_settings

################################################################################
# Set up the model

settings = hyperelasticity_settings()
n_iid_data_per_sample = 10
#I need an outer loop here on several noise stdevs:
for i, rel_noise_percent in enumerate([0.002,0.005, 0.01, 0.03, 0.1]):  #between 0.2 % and 10% $(high variance may revert to the prior, low variance may revert too easily to near-gaussian around MAP ppoint)
     #usually we use 0.01 = 1% of max = stdev, which correpsonds to very roughly 1% "SNR", to make the inference problem harder
    # can try varying amounts
    settings["rel_noise"] = rel_noise_percent
    settings["seed"] = i #different seed for each set of problems
    pde, prior, misfit, mtrue, noise_stdev = hyperelasticity_model(settings)

    model = hp.Model(pde, prior, misfit)

    Vh = pde.Vh

    ################################################################################
    # Generate training data + synthetic data observations from likelihood

    Vh = model.problem.Vh

    mesh = Vh[hp.STATE].mesh()

    B = model.misfit.B
    observable = hf.LinearStateObservable(model.problem,B)
    prior = model.prior

    dQ = 2*settings['ntargets']
    # Since the problems are
    output_basis = np.eye(dQ)

    dataGenerator = hf.DataGenerator(observable,prior, misfit) #just adding in misfit for access to the problems noise

    nsamples = 500
    data_dir = f"data/rel_noise_{rel_noise_percent}_noise_stdev_{noise_stdev}/"

    #saves m, f(m), y = f(m) + eta for several eta (n_independent_data_per_sample #)
    dataGenerator.generate(nsamples, derivatives = (1,0),output_basis = output_basis, n_data_per_sample = n_iid_data_per_sample, data_dir = data_dir)

    #resave as individual uncompressed arrays, for use with dinox

    #This is the data used by DINOX to train the dino
    mqy_data = np.load(data_dir + "mqy_data.npz")
    np.save(data_dir + "X_data.npy", mqy_data["m_data"])
    np.save(data_dir + "fX_data.npy", mqy_data["q_data"])
    np.save(
        data_dir + "dfXdX_data.npy", np.load(data_dir + "JstarPhi_data.npz")["JstarPhi_data"]
    )

    #This will be used if we need bayesian inference problems (pick an X and Y, for hte true M and true noisy data)
    np.save(data_dir + "Y_data.npy", mqy_data["y_data"])

    #Save prior parameters

    #Extract BiLaplacianPrior parameters as discrete matrices, so that we don't rely on hippylib prior instantiation afterwards
    M_mat = dl.as_backend_type(prior.M).mat()
    row,col,val = M_mat.getValuesCSR()
    M_csr = ss.csr_matrix((val,col,row))
    A_mat = dl.as_backend_type(prior.A).mat()
    row,col,val = A_mat.getValuesCSR()
    A_csr = ss.csr_matrix((val,col,row))
    sqrtMPetsc = dl.as_backend_type(prior.sqrtM).mat()
    row,col,val = sqrtMPetsc.getValuesCSR() # I think they only give the csr, so we convert
    sqrtMcsr = ss.csr_matrix((val,col,row))

    prior_params_dir = problem_dir+'prior/'
    os.makedirs(prior_params_dir, exist_ok=True)
    save_scipy_csr(prior_params_dir+"M", M_csr) #sort of redundant, each problems prior is the same, but whatever
    # print(M_csr - load_jax_csr(data_dir + "prior/M").todense(). max())
    save_scipy_csr(prior_params_dir+"sqrtM", sqrtMcsr) #sort of redundant, each problems prior is the same, but whatever
    save_scipy_csr(prior_params_dir+"A", A_csr) #sort of redundant, each problems prior is the same, but whatever
    np.save(prior_params_dir+"mean_function_coefficients.npy", prior.mean.get_local()) #sort of redundant, each problems prior is the same, but whatever

    #Save data parameters
    os.makedirs(problem_dir+'likelihood/',exist_ok=True)
    np.save(problem_dir+"likelihood/noise_stdev.npy",noise_stdev)
