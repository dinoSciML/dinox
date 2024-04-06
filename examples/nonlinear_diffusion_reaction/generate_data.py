import math
import os
import sys

import dolfin as dl
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.environ.get("HIPPYLIB_PATH"))
import hippylib as hp

sys.path.append(os.environ.get("HIPPYFLOW_PATH"))
import hippyflow as hf
# pass in the model_file_name.py
from ndr_model import (nonlinear_diffusion_reaction_model,
                       nonlinear_diffusion_reaction_settings)

# import importlib
# module = importlib.import_module(model_file_name)


################################################################################
# Set up the model
import time

start = time.time()

settings = nonlinear_diffusion_reaction_settings()
pde, prior, misfit, _ = nonlinear_diffusion_reaction_model(settings)

# pde, prior, misfit, _ = module.model(module.settings())
model = hp.Model(pde, prior, misfit)


# model =
################################################################################
# Generate training data

Vh = model.problem.Vh

mesh = Vh[hp.STATE].mesh()

B = model.misfit.B
observable = hf.LinearStateObservable(model.problem, B)
prior = model.prior

dQ = settings["ntargets"]
# Since the problems are
output_basis = np.eye(dQ)

dataGenerator = hf.DataGenerator(observable, prior)

nsamples = 25_000
data_dir = "data/"


# dataGenerator.generate(
#     nsamples, derivatives=(1, 0), output_basis=output_basis, data_dir=data_dir
# )
mq_data = np.load(data_dir + "mq_data.npz")
np.save(data_dir + "m_data.npy", mq_data["m_data"])
np.save(data_dir + "q_data.npy", mq_data["q_data"])
np.save(
    data_dir + "J_data.npy", np.load(data_dir + "JstarPhi_data.npz")["JstarPhi_data"]
)

print("total_time: ", time.time() - start)


# wrap around this saving the prior information to numpy
# warp around this saving the data generating distribution information (sigma^2)

