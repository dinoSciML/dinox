import math
import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl
import sys, os
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp
sys.path.append(os.environ.get('HIPPYFLOW_PATH'))
import hippyflow as hf

from ndr_model import nonlinear_diffusion_reaction_model, nonlinear_diffusion_reaction_settings

################################################################################
# Set up the model

settings = nonlinear_diffusion_reaction_settings()
pde, prior, misfit, _ = nonlinear_diffusion_reaction_model(settings)
model = hp.Model(pde, prior, misfit)

################################################################################
# Generate training data

Vh = model.problem.Vh

mesh = Vh[hp.STATE].mesh()

B = model.misfit.B
observable = hf.LinearStateObservable(model.problem,B)
prior = model.prior

dQ = settings['ntargets']
# Since the problems are
output_basis = np.eye(dQ)

dataGenerator = hf.DataGenerator(observable,prior)

nsamples = 1000
data_dir = 'data/'

dataGenerator.generate(nsamples, derivatives = (1,0),output_basis = output_basis, data_dir = data_dir)
