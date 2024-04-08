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
from poisson_model import *

from dinox.data_utilities import load_1D_jax_array_direct_to_gpu
import jax.experimental.sparse as jsparse
def save_scipy_csr(filename, csr_matrix): #place this where?
    filename+="_csr"
    np.save(filename+"_data", csr_matrix.data)
    np.save(filename+"_indices", csr_matrix.indices)
    np.save(filename+"_indptr", csr_matrix.indptr)
    np.save(filename+"_shape", np.array(list(csr_matrix.shape)))
def load_jax_csr(filename):
    filename+="_csr"
    data = load_1D_jax_array_direct_to_gpu(filename+"_data.npy")
    indices = load_1D_jax_array_direct_to_gpu(filename+"_indices.npy")
    ind_ptr = load_1D_jax_array_direct_to_gpu(filename+"_ind_ptr.npy")
    shape = load_1D_jax_array_direct_to_gpu(filename+"_shape.npy")
    return jsparse.CSR(data, indices, ind_ptr, shape=tuple(shape))

################################################################################
# Set up the model

formulation = "pointwise"

import time

start = time.time()
assert formulation.lower() in ["full_state", "pointwise"]

settings = poisson_settings()
model = poisson_model(settings)

################################################################################
# Generate training data

Vh = model.problem.Vh

mesh = Vh[hp.STATE].mesh()

# THIS NEEDS TO BE PART OF THE POISSON_MODEL FILE:
if formulation.lower() == "full_state":
    q_trial = dl.TrialFunction(Vh[hp.STATE])
    q_test = dl.TestFunction(Vh[hp.STATE])
    M = dl.PETScMatrix(mesh.mpi_comm())
    dl.assemble(q_trial * q_test * dl.dx, tensor=M)
    B = hf.StateSpaceIdentityOperator(M, use_mass_matrix=False)
    output_basis = None
elif formulation.lower() == "pointwise":
    B = model.misfit.B
    dQ = settings["ntargets"]
    # Since the problems are
    output_basis = np.eye(dQ)
else:
    raise

observable = hf.LinearStateObservable(model.problem, B)
prior = model.prior
for rel_noise_percent in [0.002, 0.005, 0.02, 0.2]:  #between 0.1 % and 10% $(high variance may revert to the prior low variance may revert too easily to near-gaussian around MAP ppoint)

    
    # can we make this hf.LinearStateObservable a parameter, and poisson_model a parameter
    # so that we can just run generate_data --model_dir,

    
    dataGenerator = hf.DataGenerator(observable, prior, model.misfit)
    nsamples = 25_000
    data_dir = f"data/{formulation}/rel_noise_{rel_noise_percent}+noise_stdev_{noise_stdev}/"

    dataGenerator.generate(
        nsamples, derivatives=(1, 0), output_basis=output_basis, data_dir=data_dir
    )

    #saves m, f(m), y = f(m) + eta for several eta (n_independent_data_per_sample #)
    dataGenerator.generate(nsamples, derivatives = (1,0),output_basis = output_basis, n_data_per_sample = n_iid_data_per_sample, data_dir = data_dir)

    #resave as individual uncompressed arrays, for use with dinox
    mqy_data = np.load(data_dir + "mqy_data.npz")
    np.save(data_dir + "X_data.npy", mqy_data["m_data"])
    np.save(data_dir + "fX_data.npy", mqy_data["q_data"])
    np.save(
        data_dir + "dfXdX_data.npy", np.load(data_dir + "JstarPhi_data.npz")["JstarPhi_data"]
    )
    np.save(data_dir + "Y_data.npy", mqy_data["y_data"])
    print("total_time: ", time.time() - start)


    #Also save the BiLaplacianPrior parameters as discrete matrices, so that we don't rely on hippylib prior instantiation afterwards
    M_mat = dl.as_backend_type(prior.M).mat()
    row,col,val = M_mat.getValuesCSR()
    M_csr = ss.csr_matrix((val,col,row))
    A_mat = dl.as_backend_type(prior.A).mat()
    row,col,val = A_mat.getValuesCSR()
    A_csr = ss.csr_matrix((val,col,row))
    sqrtMPetsc = dl.as_backend_type(prior.sqrtM).mat()
    row,col,val = sqrtMPetsc.getValuesCSR() # I think they only give the csr, so we convert
    sqrtMcsr = ss.csr_matrix((val,col,row))

    save_scipy_csr(data_dir + "prior/M", M_csr) #sort of redundant, each problems prior is the same, but whatever
    print(M_csr - load_jax_csr(data_dir + "prior/M").todense(). max())
    save_scipy_csr(data_dir + "prior/sqrtM.npy", sqrtMcsr) #sort of redundant, each problems prior is the same, but whatever
    save_scipy_csr(data_dir + "prior/A.npy", A_csr) #sort of redundant, each problems prior is the same, but whatever
    np.save(data_dir + "prior/mean_function_coefficients.npy", prior.mean.get_local()) #sort of redundant, each problems prior is the same, but whatever

        
