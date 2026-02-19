# Author: Lianghao Cao, Modified by Joshua Chen
# Date: 01/24/2024
# This code (the line search algoirthm) is partially provided by Tom O'Leary and written by D.C. Luo
import math
import os
import time

import dolfin as dl
import hippylib as hp
import matplotlib.pyplot as plt
import numpy as np
import ufl

from hippylib import assemblePointwiseObservation, DiscreteStateObservation

STATE, PARAMETER, Adjoint = 0, 1, 2
dl.set_log_level(dl.LogLevel.ERROR)


class ProjectedDiscreteStateObservation(DiscreteStateObservation):
    """
    This class define a misfit function for a projeted discrete linear observation operator B
    """
    
    def __init__(self, B, projector=None, data=None, noise_variance=None):
        """
        Constructor:
            :code:`B` is the observation operator   
            :code:`data` is the data
            :code:`noise_variance` is the variance of the noise
        """
        self.B = B

        if data is None:
            self.d = dl.Vector(self.B.mpi_comm())
            self.B.init_vector(self.d, 0)
        else:
            self.d = data
        
        self.PBu = dl.Vector(self.B.mpi_comm())
        self.B.init_vector(self.PBu, 0)
        self.noise_variance = noise_variance
        self.projector = projector

    def cost(self,x):
        if self.noise_variance is None: 
            raise ValueError("Noise Variance must be specified")
        elif self.noise_variance == 0:
            raise  ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
        self.B.mult(x[STATE], self.PBu)
        if self.projector is not None:
            self.PBu.set_local(self.projector @ self.PBu.get_local()) #PB u
        self.PBu.axpy(-1., self.d)
        return (.5/self.noise_variance)*self.PBu.inner(self.PBu)
    
    def grad(self, i, x, out):
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified")
        elif self.noise_variance == 0:
            raise  ZeroDivisionError("Noise Variance must not be 0.0 Set to 1.0 for deterministic inverse problems")
        if i == STATE:
            self.B.mult(x[STATE], self.PBu)
            if self.projector is not None:
                self.PBu.set_local(self.projector @ self.PBu.get_local()) #P Bu
            self.PBu.axpy(-1., self.d) #P Bu - d
            if self.projector is not None:
                self.PBu.set_local(self.PBu.get_local() @ self.projector) #P^T (P Bu - d)
            self.B.transpmult(self.PBu, out) #B^T P^T (P B u - d)/ sigma^2
            out *= (1./self.noise_variance)
        elif i == PARAMETER:
            out.zero()
        else:
            raise IndexError()
                
    def setLinearizationPoint(self,x, gauss_newton_approx=False):
        # The cost functional is already quadratic. Nothing to be done here
        return
       
    def apply_ij(self,i,j,dir,out):
        raise NotImplementedError("Not implemented")

def ProjectedPointwiseStateObservation(Vh, obs_points):
    B = assemblePointwiseObservation(Vh, obs_points)
    return ProjectedDiscreteStateObservation(B)

def settings(settings={}):
    settings["seed"] = 50  # Random seed
    settings["aspect_ratio"] = 2.0
    settings["nx"] = 80  # Number of cells in each direction

    # Prior statistics
    settings["sigma"] = 1.0  # pointwise variance
    settings["rho"] = 0.3  # spatial correlation length

    # Anisotropy of prior samples
    settings["theta0"] = 2.0
    settings["theta1"] = 0.5
    settings["alpha"] = math.atan(settings["aspect_ratio"])

    # Likelihood specs
    settings["ntargets"] = 288 
    settings["noise_variance"] = 2e-2
    settings["noise_precision"] = 50.0

    settings["dQ"] = settings["ntargets"] * 2

    # Printing and saving
    settings["verbose"] = True
    settings["output_path"] = "./result/"
    settings["plot"] = False  # plotting is broken

    # MCMC settings
    settings["k"] = 200
    return settings


def left_boundary(x, on_boundary):
    return on_boundary and (x[0] < dl.DOLFIN_EPS)


def right_boundary(x, on_boundary):
    return on_boundary and (x[0] > 2.0 - dl.DOLFIN_EPS)


class HyperelasticityVarf:

    def __init__(self):
        pass

    def parameter_map(self, m):
        return dl.Constant(3.0) * (ufl.erf(m) + dl.Constant(1.0)) + dl.Constant(1.0)

    def __call__(self, u, m, p):
        Pi = self.energy_form(u, m)
        return dl.derivative(Pi, u, p)

    def energy_form(self, u, m):
        d = u.geometric_dimension()
        Id = dl.Identity(d)
        F = Id + dl.grad(u)
        C = F.T * F

        # Lame parameters
        # -------------------
        # The new parameter-to-modulus map
        # -------------------
        E = self.parameter_map(m)
        nu = 0.4
        mu = E / (2.0 * (1.0 + nu))
        lmbda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # Invariants of the deformation tensors
        Ic, J = dl.tr(C), dl.det(F)

        # Stored strain energy density
        psi = (mu / 2.0) * (Ic - 3.0) - mu * dl.ln(J) + (lmbda / 2.0) * (dl.ln(J)) ** 2

        Pi = psi * dl.dx
        return Pi


class CustomHyperelasticityProblem(hp.PDEVariationalProblem):
    def __init__(self, Vh, stretch_length):
        bc = [
            dl.DirichletBC(
                Vh[STATE], dl.Constant((stretch_length, 0.0)), right_boundary
            ),
            dl.DirichletBC(Vh[STATE], dl.Constant((0.0, 0.0)), left_boundary),
        ]
        bc0 = [
            dl.DirichletBC(Vh[STATE], dl.Constant((0.0, 0.0)), left_boundary),
            dl.DirichletBC(Vh[STATE], dl.Constant((0.0, 0.0)), right_boundary),
        ]
        hyperelasticity_varf_hander = HyperelasticityVarf()
        super(CustomHyperelasticityProblem, self).__init__(
            Vh, hyperelasticity_varf_hander, bc, bc0, is_fwd_linear=False
        )
        u_init_expr = dl.Expression(("0.5*L*x[0]", "0.0"), L=stretch_length, degree=5)
        u_init_func = dl.interpolate(u_init_expr, Vh[STATE])
        self.u_init = u_init_func.vector()
        self.iterations = 0
        assert self.Vh[hp.STATE].mesh().mpi_comm().size == 1, print(
            "Only worked out for serial codes"
        )
        #
        # u_trial, u_test = dl.TrialFunction(Vh[STATE]), dl.TestFunction(Vh[STATE])
        # MK = dl.PETScMatrix()
        # dl.assemble(dl.inner(u_trial, u_test) * dl.dx + dl.inner(dl.grad(u_trial), dl.grad(u_test))*dl.dx, \
        #             tensor=MK)
        # self._help = self.generate_state()
        # self.MKsolver = dl.PETScLUSolver(MK)

    def parameter_map(self, m):
        m_out = self.generate_parameter()
        m_func_in = dl.Function(self.Vh[PARAMETER])
        m_func_in.vector().zero()
        m_func_in.vector().axpy(1.0, m)
        m_func_out = dl.project(
            self.varf_handler.parameter_map(m_func_in), self.Vh[PARAMETER]
        )
        m_out.axpy(1.0, m_func_out.vector())
        return m_out

    def solveFwd(self, state, x):

        # # Meets the interface condition for `solveFwd`
        # state.zero()
        # state.axpy(1., u.vector())
        # self.iterations += iteration
        x[hp.STATE].zero()
        x[hp.STATE].axpy(1.0, self.u_init)
        u = hp.vector2Function(x[hp.STATE], self.Vh[hp.STATE])
        m = hp.vector2Function(x[hp.PARAMETER], self.Vh[hp.PARAMETER])
        p = dl.TestFunction(self.Vh[hp.ADJOINT])
        F = self.varf_handler(u, m, p)
        du = dl.TrialFunction(self.Vh[hp.STATE])
        JF = dl.derivative(F, u, du)
        problem = dl.NonlinearVariationalProblem(F, u, self.bc, JF)
        solver = dl.NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm["nonlinear_solver"] = "snes"
        prm["snes_solver"]["line_search"] = "bt"
        prm["snes_solver"]["linear_solver"] = "lu"
        prm["snes_solver"]["report"] = False
        prm["snes_solver"]["error_on_nonconvergence"] = True
        prm["snes_solver"]["absolute_tolerance"] = 1e-10
        prm["snes_solver"]["relative_tolerance"] = 1e-6
        prm["snes_solver"]["maximum_iterations"] = 1000
        prm["newton_solver"]["absolute_tolerance"] = 1e-10
        prm["newton_solver"]["relative_tolerance"] = 1e-6
        prm["newton_solver"]["maximum_iterations"] = 1000
        prm["newton_solver"]["relaxation_parameter"] = 1.0

        # print(dl.info(solver.parameters, True))
        iterations, converged = solver.solve()
        
        self.iterations += iterations
        state.zero()
        state.axpy(1.0, u.vector())


def HyperelasticityPrior(
    Vh_PARAMETER, pointwise_std, correlation_length, mean=None, anis_diff=None
):
    # Delta and gamma
    delta = 1.0 / (pointwise_std * correlation_length)
    gamma = delta * correlation_length**2
    if anis_diff is None:
        theta0 = 1
        theta1 = 1
        alpha = math.pi / 4.0
        anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree=1)
        anis_diff.set(theta0, theta1, alpha)
    if mean is None:
        return hp.BiLaplacianPrior(Vh_PARAMETER, gamma, delta, anis_diff, robin_bc=True)
    else:
        return hp.BiLaplacianPrior(
            Vh_PARAMETER, gamma, delta, anis_diff, mean=mean, robin_bc=True
        )


def HyperelasticityMisfit(Vu, settings):
    # Create misfit object
    ndim = 2
    # Uniform target
    ntargets = settings["ntargets"]
    aspect_ratio = settings["aspect_ratio"]

    ny_targets = math.floor(math.sqrt(ntargets / aspect_ratio))
    nx_targets = math.floor(ntargets / ny_targets)
    if not ny_targets * nx_targets == ntargets:
        raise Exception(
            "The number of observation points cannot lead to a regular grid \
        that is compatible with the aspect ratio."
        )
    targets_x = np.linspace(0.0, settings["aspect_ratio"], nx_targets + 2)
    targets_y = np.linspace(0.0, 1.0, ny_targets + 2)
    targets_xx, targets_yy = np.meshgrid(targets_x[1:-1], targets_y[1:-1])
    targets = np.zeros([ntargets, ndim])
    targets[:, 0] = targets_xx.flatten()
    targets[:, 1] = targets_yy.flatten()
    settings["targets"] = targets
    
    misfit = ProjectedPointwiseStateObservation(Vu, targets)

    misfit.noise_variance = settings["noise_variance"]
    misfit.noise_precision = settings["noise_precision"]
    return misfit, targets 


def model(settings):
    np.random.seed(settings["seed"])
    output_path = settings["output_path"]

    if output_path is not None and not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    ndim = 2
    nx = settings["nx"]
    mesh = dl.RectangleMesh(
        dl.Point(0, 0),
        dl.Point(settings["aspect_ratio"], 1.0),
        nx,
        math.floor(nx / settings["aspect_ratio"]),
    )
    Vh_STATE = dl.VectorFunctionSpace(mesh, "Lagrange", 2)
    Vh_PARAMETER = dl.FunctionSpace(mesh, "Lagrange", 1)
    Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE]

    pde = CustomHyperelasticityProblem(Vh, 1.5)

    theta0 = settings["theta0"]
    theta1 = settings["theta1"]
    alpha = settings["alpha"]
    anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree=5)
    anis_diff.set(theta0, theta1, alpha)
    prior = HyperelasticityPrior(
        Vh[PARAMETER], settings["sigma"], settings["rho"], anis_diff=anis_diff
    )

    misfit, targets  = HyperelasticityMisfit(Vh[STATE], settings
    )

    return hp.Model(
        pde,
        prior,
        misfit
    )

