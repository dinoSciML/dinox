# This document is ALMOST ENTIRELY based on the hippyflow helmhotz 2d problem.
# Please cite hippyflow if you use this part of the code.
# import logging
import math
import os

import dolfin as dl
import hippylib as hp
import numpy as np
import ufl

# from .visual_tool import plot_component
# import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=20)
# import matplotlib
# matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

STATE, PARAMETER, ADJOINT = 0, 1, 2

from hippylib import (DiscreteStateObservation, PDEProblem,
                      assemblePointwiseObservation)


def settings(settings={}):
    settings["seed"] = 0  # Random seed
    settings["nx"] = 64  # Number of cells in each direction
    settings["frequency"] = 300

    # Prior statistics
    settings["sigma"] = 0.45  # pointwise variance
    settings["rho"] = 0.45  # spatial correlation length

    # Anisotropy of prior samples
    settings["theta0"] = 1.0
    settings["theta1"] = 1.0
    settings["alpha"] = 0.25 * math.pi

    # Likelihood specs
    settings["ntargets"] = 16
    # settings["rel_noise"] = 0.02
    settings["rel_noises"] = [
        0.002,
        0.01,
    ]  # used by generate_data.py
    settings["dQ"] = settings["ntargets"]

    # Printing and saving
    settings["verbose"] = True
    settings["output_path"] = "./figures"
    settings["plot"] = False
    # MCMC settings
    settings["number_of_samples"] = 4000
    settings["output_frequency"] = 50
    settings["step_size"] = 0.1
    settings["burn_in"] = 500
    settings["k"] = 16
    settings["p"] = 20

    return settings


class ComponentObservationOperator:
    def __init__(self, Vh, obs_points, B, component):
        self.B = B
        self.outdim = obs_points.shape[-1]
        self.ntargets = obs_points.shape[0]
        self.Vo = dl.VectorFunctionSpace(Vh.mesh(), "R", degree=0, dim=obs_points.shape[0])
        trial = dl.TrialFunction(self.Vo)
        test = dl.TestFunction(self.Vo)
        # Identity mass matrix
        self.M = dl.assemble(ufl.inner(trial, test) * ufl.dx)
        self._out_full = dl.Vector()
        self.B.init_vector(self._out_full, 0)
        if component >= self.outdim:
            raise Exception("The component exceed dimension of the state function value")
        self.component = component

    def mpi_comm(self):
        return self.M.mpi_comm()

    def init_vector(self, x, dim):
        self.M.init_vector(x, dim)

    def mult(self, x, y):
        y.zero()
        self._out_full.zero()
        self.B.mult(x, self._out_full)
        full_obs = self._out_full.get_local().reshape(self.ntargets, self.outdim)
        y.set_local(full_obs[:, self.component])

    def transpmult(self, x, y):
        y.zero()
        full_obs = np.zeros((self.ntargets, self.outdim))
        full_obs[:, self.component] = x.get_local()
        self._out_full.set_local(full_obs.flatten())
        self.B.transpmult(self._out_full, y)


class ComponentPointwiseStateObservation(DiscreteStateObservation):
    def __init__(self, Vh, obs_points, component=0):
        B = assemblePointwiseObservation(Vh, obs_points)
        self.B = ComponentObservationOperator(Vh, obs_points, B, component)
        self.d = dl.Vector(Vh.mesh().mpi_comm())
        self.B.init_vector(self.d, 0)
        self.Bu = dl.Vector(Vh.mesh().mpi_comm())
        self.B.init_vector(self.Bu, 0)
        self.noise_variance = None


class PML:
    def __init__(self, mesh, box, box_pml, A):

        t = [None] * 4

        for i in range(4):
            t[i] = box_pml[i] - box[i]
            if abs(t[i]) < dl.DOLFIN_EPS:
                t[i] = 1.0

        self.sigma_x = dl.Expression(
            "(x[0]<xL)*A*(x[0]-xL)*(x[0]-xL)/(tL*tL) + (x[0]>xR)*A*(x[0]-xR)*(x[0]-xR)/(tR*tR)",
            xL=box[0],
            xR=box[2],
            A=A,
            tL=t[0],
            tR=t[2],
            degree=2,
        )

        self.sigma_y = dl.Expression(
            "(x[1]<yB)*A*(x[1]-yB)*(x[1]-yB)/(tB*tB) + (x[1]>yT)*A*(x[1]-yT)*(x[1]-yT)/(tT*tT)",
            yB=box[1],
            yT=box[3],
            A=A,
            tB=t[1],
            tT=t[3],
            degree=2,
        )

        physical_domain = dl.AutoSubDomain(
            lambda x, on_boundary: x[0] >= box[0] and x[0] <= box[2] and x[1] >= box[1] and x[1] <= box[3]
        )

        cell_marker = dl.MeshFunction("size_t", mesh, mesh.geometry().dim())
        cell_marker.set_all(0)
        physical_domain.mark(cell_marker, 1)
        self.dx = dl.Measure("dx", subdomain_data=cell_marker)


class SingleSourceHelmholtzProblem(PDEProblem):
    def __init__(self, Vh, sources_loc, wave_number, pml):
        self.Vh = Vh
        self.wave_number = wave_number
        self.PML = pml

        self.rhs_fwd = self.generate_state()

        if type(sources_loc) is dl.Point:
            ps0 = dl.PointSource(self.Vh[STATE].sub(0), sources_loc, 1.0)
            ps0.apply(self.rhs_fwd)

        else:
            for source in sources_loc:
                ps0 = dl.PointSource(self.Vh[STATE].sub(0), source, 1.0)
                ps0.apply(self.rhs_fwd)

        self.A = None
        self.At = None
        self.C = None
        self.Wmu = None
        self.Wmm = None
        self.Wuu = None

        self.solver = self._createLUSolver()
        self.solver_fwd_inc = self._createLUSolver()
        self.solver_adj_inc = self._createLUSolver()

    def varf_handler(self, u, m, p):

        k = self.wave_number * dl.exp(m)
        ksquared = k**2

        sigma_x = self.PML.sigma_x
        sigma_y = self.PML.sigma_y

        Kr = ksquared - sigma_x * sigma_y
        Ki = -k * (sigma_x + sigma_y)

        Dr_xx = (ksquared + sigma_x * sigma_y) / (ksquared + sigma_x * sigma_x)
        Dr_yy = (ksquared + sigma_x * sigma_y) / (ksquared + sigma_y * sigma_y)
        Di_xx = k * (sigma_x - sigma_y) / (ksquared + sigma_x * sigma_x)
        Di_yy = k * (sigma_y - sigma_x) / (ksquared + sigma_y * sigma_y)

        Dr = dl.as_matrix([[Dr_xx, dl.Constant(0.0)], [dl.Constant(0.0), Dr_yy]])
        Di = dl.as_matrix([[Di_xx, dl.Constant(0.0)], [dl.Constant(0.0), Di_yy]])

        u1, u2 = dl.split(u)
        p1, p2 = dl.split(p)

        form_r = dl.inner(dl.grad(u1), dl.grad(p1)) * self.PML.dx(1) - ksquared * u1 * p1 * self.PML.dx(1)

        form_i = -dl.inner(dl.grad(u2), dl.grad(p2)) * self.PML.dx(1) + ksquared * u2 * p2 * self.PML.dx(1)

        form_pml_r = (
            dl.inner(Dr * dl.grad(u1), dl.grad(p1)) * self.PML.dx(0)
            + dl.inner(Di * dl.grad(u2), dl.grad(p1)) * self.PML.dx(0)
            - Kr * u1 * p1 * self.PML.dx(0)
            - Ki * u2 * p1 * self.PML.dx(0)
        )

        form_pml_i = (
            -dl.inner(Dr * dl.grad(u2), dl.grad(p2)) * self.PML.dx(0)
            + dl.inner(Di * dl.grad(u1), dl.grad(p2)) * self.PML.dx(0)
            + Kr * u2 * p2 * self.PML.dx(0)
            - Ki * u1 * p2 * self.PML.dx(0)
        )

        return form_r + form_i + form_pml_r + form_pml_i

    def generate_state(self):
        """return a vector in the shape of the state"""
        return dl.Function(self.Vh[STATE]).vector()

    def generate_parameter(self):
        """return a vector in the shape of the parameter"""
        return dl.Function(self.Vh[PARAMETER]).vector()

    def init_parameter(self, m):
        """initialize the parameter"""
        dummy = self.generate_parameter()
        m.init(dummy.mpi_comm(), dummy.local_range())

    def solveFwd(self, state, x):
        """Solve the possibly nonlinear Fwd Problem:
        Given m, find u such that
        \delta_p F(u,m,p;\hat_p) = 0 \for all \hat_p"""

        u = dl.TrialFunction(self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = dl.TestFunction(self.Vh[ADJOINT])
        A = dl.assemble(self.varf_handler(u, m, p))

        self.solver.set_operator(A)
        self.solver.solve(state, self.rhs_fwd)

    def solveAdj(self, adj, x, adj_rhs):
        """Solve the linear Adj Problem:
        Given m, u; find p such that
        \delta_u F(u,m,p;\hat_u) = 0 \for all \hat_u
        """
        u = dl.TestFunction(self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = dl.TrialFunction(self.Vh[ADJOINT])
        Aadj = dl.assemble(self.varf_handler(u, m, p))

        self.solver.set_operator(Aadj)
        self.solver.solve(adj, adj_rhs)

    def evalGradientParameter(self, x, out):
        """Given u,m,p; eval \delta_m F(u,m,p; \hat_m) \for all \hat_m"""
        u = hp.vector2Function(x[STATE], self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = hp.vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        dm = dl.TestFunction(self.Vh[PARAMETER])
        res_form = self.varf_handler(u, m, p)
        out.zero()
        dl.assemble(dl.derivative(res_form, m, dm), tensor=out)

    def setLinearizationPoint(self, x, gn_approx):
        """Set the values of the state and parameter
        for the incremental Fwd and Adj solvers"""
        u = hp.vector2Function(x[STATE], self.Vh[STATE])
        m = hp.vector2Function(x[PARAMETER], self.Vh[PARAMETER])
        p = hp.vector2Function(x[ADJOINT], self.Vh[ADJOINT])
        x_fun = [u, m, p]

        f_form = self.varf_handler(u, m, p)

        g_form = [None, None, None]
        for i in range(3):
            g_form[i] = dl.derivative(f_form, x_fun[i])

        self.A = dl.assemble(dl.derivative(g_form[ADJOINT], u))
        self.At = dl.assemble(dl.derivative(g_form[STATE], p))
        self.C = dl.assemble(dl.derivative(g_form[ADJOINT], m))

        self.solver_fwd_inc.set_operator(self.A)
        self.solver_adj_inc.set_operator(self.At)

        if gn_approx:
            self.Wmu = None
            self.Wuu = None
            self.Wmm = None
        else:
            self.Wmu = dl.assemble(dl.derivative(g_form[PARAMETER], u))
            self.Wuu = dl.assemble(dl.derivative(g_form[STATE], u))
            self.Wmm = dl.assemble(dl.derivative(g_form[PARAMETER], m))

    def solveIncremental(self, out, rhs, is_adj):
        """If is_adj = False:
        Solve the forward incremental system:
        Given u, m, find \tilde_u s.t.:
        \delta_{pu} F(u,m,p; \hat_p, \tilde_u) = rhs for all \hat_p.

        If is_adj = True:
        Solve the adj incremental system:
        Given u, m, find \tilde_p s.t.:
        \delta_{up} F(u,m,p; \hat_u, \tilde_p) = rhs for all \delta_u.
        """
        if is_adj:
            self.solver_adj_inc.solve(out, rhs)
        else:
            self.solver_fwd_inc.solve(out, rhs)

    def apply_ij(self, i, j, dir, out):
        """
        Given u, m, p; compute
        \delta_{ij} F(u,a,p; \hat_i, \tilde_j) in the direction \tilde_j = dir for all \hat_i
        """
        KKT = {}
        KKT[STATE, STATE] = self.Wuu
        KKT[PARAMETER, STATE] = self.Wmu
        KKT[PARAMETER, PARAMETER] = self.Wmm
        KKT[ADJOINT, STATE] = self.A
        KKT[ADJOINT, PARAMETER] = self.C

        if i >= j:
            KKT[i, j].mult(dir, out)
        else:
            KKT[j, i].transpmult(dir, out)

    def _createLUSolver(self):
        if hp.dlversion() <= (1, 6, 0):
            return hp.PETScLUSolver()
        else:
            return hp.PETScLUSolver(self.Vh[STATE].mesh().mpi_comm())


def model(settings):
    np.random.seed(settings["seed"])
    output_path = settings["output_path"]

    if output_path is not None and not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    box = [0.0, 0.0, 3.0, 3.0]
    box_pml = [-1.0, -1.0, 4.0, 3.0]
    nx = settings["nx"]
    mesh = dl.RectangleMesh(
        dl.Point(box_pml[0], box_pml[1]),
        dl.Point(box_pml[2], box_pml[3]),
        nx,
        nx,
    )
    Vh2 = dl.VectorFunctionSpace(mesh, "Lagrange", 2)
    Vh1 = dl.FunctionSpace(mesh, "Lagrange", 1)
    Vh = [Vh2, Vh1, Vh2]
    dims = [Vhi.dim() for Vhi in Vh]

    source_loc_ = ((box[0] + 0.1 + (box[2] - 0.1) / 2) / 2, box[3] - 0.15)

    sources_loc = [dl.Point(*source_loc_)]

    c = 343.4  # m/s     speed of sound in air
    rho = 1.204  # kg/m^3  density of air
    all_frequencies = np.array([settings["frequency"]])
    pml = PML(mesh, box, box_pml, 50)

    f = all_frequencies[0]
    omega = 2.0 * np.pi * f
    wave_number = dl.Constant(omega / (c * rho))
    pde = SingleSourceHelmholtzProblem(Vh, sources_loc, wave_number, pml)

    sigma = settings["sigma"]
    rho = settings["rho"]
    delta = 1.0 / (sigma * rho)
    gamma = delta * rho**2

    theta0 = settings["theta0"]
    theta1 = settings["theta1"]
    alpha = settings["alpha"]

    anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree=1)
    anis_diff.set(theta0, theta1, alpha)

    prior = hp.BiLaplacianPrior(Vh[PARAMETER], gamma, delta, anis_diff, robin_bc=True)

    ndim = 2
    obs_length = 0.2

    rel_noise = settings.get("rel_noise")

    ntargets = settings["ntargets"]
    x_targets = np.linspace(
        box[0] + 0.05 * (box[2] - box[0]),
        box[2] - 0.05 * (box[2] - box[0]),
        ntargets + 1,
    )
    idx = (np.abs(x_targets - source_loc_[0])).argmin()
    x_targets = np.delete(x_targets, idx)
    # y_targets = np.linspace(box[3] - 0.05 - obs_length, box[3] - obs_length + 0.15, math.floor(np.sqrt(ntargets)))
    targets = []
    for xi in x_targets:
        targets.append((xi, box[3] - 0.05))
    targets = np.array(targets)
    settings["targets"] = targets
    misfit = ComponentPointwiseStateObservation(Vh[STATE], targets, component=0)

    if os.path.isfile(output_path + "inverse_problem_data.npz"):
        if settings["verbose"]:
            print("Loading observation data")
        ip_data = np.load(output_path + "inverse_problem_data.npz")
        d_data = ip_data["d_data"]
        noise_std_dev = ip_data["noise_std_dev"]
        misfit.d.set_local(d_data)
    else:
        if settings["verbose"]:
            print("Generating observation data")
        # utrue = pde.generate_state()
        # mtrue = true_parameter(prior)
        # x = [utrue, mtrue, None]
        # pde.solveFwd(x[hp.STATE], x)
        # misfit.B.mult(x[hp.STATE], misfit.d)
        # MAX = misfit.d.norm("linf")
        # noise_std_dev = rel_noise * MAX
        # hp.parRandom.normal_perturb(noise_std_dev, misfit.d)
        # if settings["verbose"]:
        #     print("Saving observation data")
        utrue = pde.generate_state()

        if rel_noise is not None:
            noise_std_dev = 0.0
            print("Determining noise standard deviation choice based on 50 random samples")
            for i in range(50):
                utrue = pde.generate_state()
                mtrue = true_parameter(prior)  # random true parameter
                x = [utrue, mtrue, None]
                pde.solveFwd(x[hp.STATE], x)
                misfit.B.mult(x[hp.STATE], misfit.d)
                MAX = misfit.d.norm("linf")
                noise_std_dev = max(rel_noise * MAX, noise_std_dev)
            hp.parRandom.normal_perturb(noise_std_dev, misfit.d)
            misfit.noise_variance = noise_std_dev * noise_std_dev
            print("noise_std_dev =", noise_std_dev)
        else:
            noise_std_dev = None
            mtrue = None
        if settings["plot"] and output_path is not None:
            d_data = misfit.d.get_local()
            # plot_component(Vh[STATE], utrue, component=0)
            # cbar = plt.scatter(targets[:,0], targets[:,1], c= d_data, marker=",", s=10)
            # plt.colorbar(cbar)
            # plt.xticks([])
            # plt.yticks([])
            # plt.xlim([box[0],box[2]])
            # plt.ylim([box[1],box[3]])
            # plt.gca().set_aspect('equal')
            # plt.savefig(output_path + "obs_sample.pdf", bbox_inches="tight")
            # plt.close()
            # m_data = mtrue.get_local()
            # np.savez(output_path + "inverse_problem_data.npz",d_data = d_data,\
            #                     noise_std_dev = noise_std_dev,m_data = m_data)

    return pde, prior, misfit, mtrue, noise_std_dev


def true_parameter(prior):
    noise = dl.Vector()
    prior.init_vector(noise, "noise")
    hp.parRandom.normal(1.0, noise)
    mtrue = dl.Vector()
    prior.init_vector(mtrue, 0)
    prior.sample(noise, mtrue)
    return mtrue
