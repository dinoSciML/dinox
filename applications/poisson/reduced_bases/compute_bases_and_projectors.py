import os, sys
import dolfin as dl
import math
import ufl
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla 
import scipy.linalg 

import matplotlib.pyplot as plt
import math
import time

sys.path.append( os.environ.get('HIPPYLIB_PATH', "...") )
import hippylib as hp

sys.path.append( os.environ.get('HIPPYFLOW_PATH'))
import hippyflow as hf

sys.path.append('../')

from poisson_model import *

################################################################################
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', '--data_dir', type=str, default='../data/pointwise/', help="Where to save")
parser.add_argument('-basis_type', '--basis_type', type=str, default='as', help="pod, as or kle")
parser.add_argument('-rank', '--rank', type=int, default=400, help="Active subspace rank")
parser.add_argument('-oversample', '--oversample', type=int, default=10, help="Active subspace oversample")
parser.add_argument('-ndata', '--ndata', type=int, default=800, help="Number of samples")

args = parser.parse_args()

data_dir = args.data_dir

################################################################################
# Parameters

rank = args.rank
oversample = args.oversample

################################################################################
# Set up the model
import time

start = time.time()
start0 = start
settings = poisson_settings()
model = poisson_model(settings)

Vh = model.problem.Vh

#Only NEED TO LOAD THE PRIOR R csr_matrix!
prior = model.prior
print("time to load prior:", time.time()-start)
assert dl.MPI.comm_world.size == 1, print('Not thought out in other cases yet')


if args.basis_type.lower() == 'pod':
	if 'pointwise' in args.data_dir:
		print('This POD is meant for full state. Better luck next time!!')
		raise 
	data_dir = args.data_dir
	all_data = np.load(data_dir+'mq_data.npz')

	u_data = all_data['q_data'][:args.ndata]

	POD = hf.PODProjectorFromData(Vh)

	d_POD, phi, Mphi, u_shift = POD.construct_subspace(u_data,rank)

	check_orth = True
	if check_orth:
		PsistarPsi = Mphi.T@phi
		orth_error = np.linalg.norm(PsistarPsi - np.eye(PsistarPsi.shape[0]))
		print('||Psi^*Psi - I|| = ',orth_error)
		assert orth_error < 1e-5

	np.save('POD_basis',phi)
	np.save('POD_d',d_POD)
	np.save('POD_projector',Mphi)
	np.save('POD_shift',u_shift)

	fig, ax = plt.subplots()
	ax.semilogy(np.arange(len(d_POD)), d_POD)

	ax.set(xlabel='index', ylabel='eigenvalue',
		   title='POD spectrum')
	ax.grid()

	fig.savefig("POD_eigenvalues.pdf")

elif args.basis_type.lower() == 'kle':
	KLE = hf.KLEProjector(prior)
	KLE.parameters['rank'] = rank
	KLE.parameters['oversampling'] = oversample
	KLE.parameters['save_and_plot'] = False

	d_KLE, kle_basis, kle_projector = KLE.construct_input_subspace()

	input_basis = hf.mv_to_dense(kle_basis)
	input_projector = hf.mv_to_dense(kle_projector)


	check_orth = True
	if check_orth:
		PsistarPsi = input_projector.T@input_basis
		orth_error = np.linalg.norm(PsistarPsi - np.eye(PsistarPsi.shape[0]))
		print('||Psi^*Psi - I|| = ',orth_error)
		assert orth_error < 1e-5

	np.save('KLE_basis',input_basis)
	np.save('KLE_d',d_KLE)
	np.save('KLE_projector',input_projector)

elif args.basis_type.lower() == 'as':
	################################################################################
	# Load the data

	data_dir = args.data_dir
	all_data = np.load(data_dir+'mq_data.npz')
	# all_data = np.load(data_dir+'mq_data.npz')
	JTPhi_data = np.load(data_dir+'JstarPhi_data.npz')['JstarPhi_data'][:args.ndata]

	m_data = all_data['m_data'][:args.ndata]
	q_data = all_data['q_data'][:args.ndata]
	# PhiTJ_data = np.transpose(JTPhi_data['JstarPhi_data'], (0,2,1))[:args.ndata] #TODO, remove need to transpose!

	print('m_data.shape = ',m_data.shape)
	print('q_data.shape = ',q_data.shape)
	print('PhiTJ_data.shape = ',JTPhi_data.shape)


	################################################################################
	# Instance JTJ operator 
	# print('Loading JTJ')
	new_approach = True
	t0 = time.time()
	if not new_approach:
		PhiTJ_data = np.transpose(JTPhi_data, (0,2,1))
		JTJ_operator = hf.MeanJTJfromDataOperator(PhiTJ_data, prior)
		# Set up the Gaussian random
		m_vector = dl.Vector()
		JTJ_operator.init_vector_lambda(m_vector,0)
		Omega = hp.MultiVector(m_vector,rank+oversample)
		hp.parRandom.normal(1.,Omega)
	
	else:
		dM = m_data.shape[1]
		OmegaNumpy = np.random.normal(size=(dM, rank+oversample))
		Mcsr, sqrtMcsr, A_csr, mean = prior._get_parameters()
		print("Creating R sparse, only do once, for all time, save to disk, maybe just load from disk")
		start = time.time()
		# R_sparse =  A_csr.T@scipy.sparse.linalg.spsolve(Mcsr.tocsc(),A_csr.tocsc())
		# print("forming R sparse time", time.time()-start)
		# print(type(R_sparse))
		Minv = spla.inv(Mcsr)
		print("forming M_inv sparse time (will be loading from disk instead)", time.time()-start)

		# print(np.linalg.norm(Minv@Mcsr - np.eye(dM)))
		# start = time.time()
		R_sparse =  A_csr.T@Minv@A_csr
		#M_inv is of the order of the same size as 
		print("forming R sparse time, will just load from disk isnead", time.time()-start)
		# print(Minv.nnz, dM*dM, R_sparse.nnz)


		# A_csr.T@scipy.sparse.linalg.spsolve(Mcsr@A_csr
		# R_sparse_action 
		if False:
			R_sparse_or_dense = R_dense
		else:
			R_sparse_or_dense = R_sparse

	print('Beginning doublePassG')

	if hasattr(prior, "R"): #TODO: check if the JTJ action can be sped up by creating a Omega numpy Matrix, rather than Multivector (since here J.T J is dense actions)
		if new_approach:
			d_GN, V_GN = hp.doublePassGAATNumpy(JTPhi_data,prior.R, prior, 
				R_sparse_or_dense, OmegaNumpy,rank,s=1)
		else:
			d_GN, V_GN = hp.doublePassG(JTJ_operator,prior.R, prior.Rsolver, Omega,rank,s=1)
	else:
		raise
		# d_GN, V_GN = hp.doublePassGAATNumpy(JTJ_operator,\
		# 	prior.Hlr, prior.Hlr, OmegaNumpy,rank,s=1)

	print('doublePassGAATNumpy took ',time.time() - t0,'s')

	input_basis = hf.mv_to_dense(V_GN)

	# Compute the projector RV_r from the basis
	RV_GN = hp.MultiVector(V_GN[0],V_GN.nvec())
	RV_GN.zero()
	hp.MatMvMult(prior.R,V_GN,RV_GN) #R is C^-1

	input_projector = hf.mv_to_dense(RV_GN)
	#dense_to_mv

	check_orth = True
	if check_orth:
		PsistarPsi = input_projector.T@input_basis
		orth_error = np.linalg.norm(PsistarPsi - np.eye(PsistarPsi.shape[0]))
		print('||Psi^*Psi - I|| = ',orth_error)
		assert orth_error < 1e-5

	np.save('AS_input_basis',input_basis)
	np.save('AS_d_GN',d_GN)
	np.save('AS_input_projector',input_projector)

	fig, ax = plt.subplots()
	ax.semilogy(np.arange(len(d_GN)), d_GN)

	ax.set(xlabel='index', ylabel='eigenvalue',
		   title='GEVP JJT spectrum')
	ax.grid()

	fig.savefig("JJT_eigenvalues.pdf")

else: 
	raise



print("Total runtime:", time.time()-start0)



