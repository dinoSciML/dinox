import math
import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl
import sys, os
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp
sys.path.append(os.environ.get('HIPPYFLOW_PATH'))
import hippyflow as hf

from poisson_model import *

################################################################################
# Set up the model

formulation = 'pointwise'

import time
start = time.time()
assert formulation.lower() in ['full_state', 'pointwise']

settings = poisson_settings()
model = poisson_model(settings)

################################################################################
# Generate training data

Vh = model.problem.Vh

mesh = Vh[hp.STATE].mesh()

if formulation.lower() == 'full_state':
	q_trial = dl.TrialFunction(Vh[hp.STATE])
	q_test = dl.TestFunction(Vh[hp.STATE])
	M = dl.PETScMatrix(mesh.mpi_comm())
	dl.assemble(q_trial * q_test * dl.dx, tensor=M)
	B = hf.StateSpaceIdentityOperator(M, use_mass_matrix=False)
	output_basis = None
elif formulation.lower() == 'pointwise':
	B = model.misfit.B
	dQ = settings['ntargets']
	# Since the problems are
	output_basis = np.eye(dQ)
else:
	raise

observable = hf.LinearStateObservable(model.problem,B)
prior = model.prior

dataGenerator = hf.DataGenerator(observable,prior)

nsamples = 1000
data_dir = 'data/'+formulation+'/'

dataGenerator.generate(nsamples, derivatives = (1,0),output_basis = output_basis, data_dir = data_dir)

print("total_time: ", time.time()-start)