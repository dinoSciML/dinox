# Author: Lianghao Cao
# Date: 01/24/2024
# This code (the line search algoirthm) is partially provided by Tom O'Leary and written by D.C. Luo
import numpy as np
import math
import os, ufl
import dolfin as dl
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
# from scipy.sparse import csc_matrix, csr_matrix
# from scipy.sparse import linalg as spla
import hippylib as hp
import time

STATE, PARAMETER, Adjoint = 0, 1, 2
dl.set_log_level(dl.LogLevel.ERROR)


def settings(settings={}):
    settings["seed"] = 50  # Random seed
    settings["aspect_ratio"] = 2.0
    settings["nx"] = 64  # Number of cells in each direction

    # Prior statistics
    settings["sigma"] = 1.0  # pointwise variance
    settings["rho"] = 0.3  # spatial correlation length

    # Anisotropy of prior samples
    settings["theta0"] = 2.0
    settings["theta1"] = 0.5
    settings["alpha"] = math.atan(settings["aspect_ratio"])

    # Likelihood specs
    settings["ntargets"] = 32
    settings["rel_noises"] = [0.002, 0.005, 0.01, 0.02, 0.05] #used by generate_data.py
    settings["dQ"] = settings["ntargets"]*2


    # Printing and saving
    settings["verbose"] = True
    settings["output_path"] = "./result/"
    settings["plot"] = False #plotting is broken

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
        bc = [dl.DirichletBC(Vh[STATE], dl.Constant((stretch_length, 0.0)), right_boundary), \
              dl.DirichletBC(Vh[STATE], dl.Constant((0.0, 0.0)), left_boundary)]
        bc0 = [dl.DirichletBC(Vh[STATE], dl.Constant((0.0, 0.0)), left_boundary), \
               dl.DirichletBC(Vh[STATE], dl.Constant((0.0, 0.0)), right_boundary)]
        hyperelasticity_varf_hander = HyperelasticityVarf()
        super(CustomHyperelasticityProblem, self).__init__(Vh, hyperelasticity_varf_hander, bc, bc0,
                                                           is_fwd_linear=False)
        u_init_expr = dl.Expression(("0.5*L*x[0]", "0.0"), L=stretch_length, degree=5)
        u_init_func = dl.interpolate(u_init_expr, Vh[STATE])
        self.u_init = u_init_func.vector()
        self.iterations = 0
        assert self.Vh[hp.STATE].mesh().mpi_comm().size == 1, print('Only worked out for serial codes')
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
        m_func_in.vector().axpy(1., m)
        m_func_out = dl.project(self.varf_handler.parameter_map(m_func_in), self.Vh[PARAMETER])
        m_out.axpy(1., m_func_out.vector())
        return m_out

    def solveFwd(self, state, x):

        # u = dl.Function(self.Vh[STATE])
        # m = hp.vector2Function(x[hp.PARAMETER], self.Vh[hp.PARAMETER])
        # p = dl.TestFunction(self.Vh[hp.ADJOINT])
        # F = self.varf_handler(u, m, p)
        # du = dl.TrialFunction(self.Vh[hp.STATE])
        # JF = dl.derivative(F, u, du)
        # A_dl, b_dl = dl.assemble_system(JF, F, bcs=self.bc)  # bc is the state variable bc
        # solver = dl.PETScLUSolver()
        # solver.solve(A_dl, u.vector(), -b_dl)
        # energy_old = dl.assemble(self.varf_handler.energy_form(u, m))
        #
        # # Parameters for the problem
        # tolerance = 1e-8
        # max_iterations = 200
        # # Line search
        # line_search = True
        # max_backtrack = 10
        # alpha0 = 1.0
        # # Printing
        # verbose = True
        #
        # # Initialize while loop variables
        # converged = False
        # iteration = 1
        # u_old = self.generate_state()
        # delta_u = self.generate_state()
        # while not converged:
        #     iteration += 1
        #     # Assemble A and enforce BCS
        #     A_dl, b_dl = dl.assemble_system(JF, F, bcs=self.bc0)  # bc0 is the zero Dirichlet BC
        #     # Compute residual norm
        #     self.MKsolver.solve(self._help, b_dl)
        #     residual_norm = b_dl.inner(self._help)
        #     if verbose:
        #         print('At iteration ', iteration, 'the residual norm = ', residual_norm)
        #     converged = (residual_norm < tolerance)
        #
        #     solver.solve(A_dl, delta_u, -b_dl)
        #
        #     if line_search:
        #         u_old.zero()
        #         u_old.axpy(1., u.vector())
        #         alpha = alpha0
        #         backtrack_iteration = 0
        #         searching = True
        #         while searching:
        #             backtrack_iteration += 1
        #             u.vector().axpy(alpha, delta_u)
        #             energy_new = dl.assemble(self.varf_handler.energy_form(u, m))
        #             if energy_new <= energy_old + 1.0e-4 * alpha * b_dl.inner(delta_u):
        #                 energy_old = energy_new
        #                 searching = False
        #             else:
        #                 if verbose:
        #                     print("energy: ", energy_new)
        #                     print('Need to take a smaller step for alpha = ', alpha)
        #                 u.vector().zero()
        #                 u.vector().axpy(1., u_old)
        #                 alpha *= 0.5
        #             if backtrack_iteration > max_backtrack:
        #                 raise Exception("Backtrack failed")
        #     else:
        #         u.vector().axpy(1., delta_u)
        #
        #     if iteration > max_iterations:
        #         print('Maximum iterations for nonlinear PDE solve reached, moving on.')
        #         print('Final residual norm = ', residual_norm)
        #         break
        #
        # # Meets the interface condition for `solveFwd`
        # state.zero()
        # state.axpy(1., u.vector())
        # self.iterations += iteration
        x[hp.STATE].zero()
        x[hp.STATE].axpy(1., self.u_init)
        u = hp.vector2Function(x[hp.STATE], self.Vh[hp.STATE])
        m = hp.vector2Function(x[hp.PARAMETER], self.Vh[hp.PARAMETER])
        p = dl.TestFunction(self.Vh[hp.ADJOINT])
        F = self.varf_handler(u, m, p)
        du = dl.TrialFunction(self.Vh[hp.STATE])
        JF = dl.derivative(F, u, du)
        problem = dl.NonlinearVariationalProblem(F, u, self.bc, JF)
        solver = dl.NonlinearVariationalSolver(problem)

        prm = solver.parameters
        prm['nonlinear_solver'] = 'snes'
        prm['snes_solver']['line_search'] = 'bt'
        prm['snes_solver']['linear_solver'] = 'lu'
        prm['snes_solver']['report'] = False
        prm['snes_solver']['error_on_nonconvergence'] = True
        prm['snes_solver']['absolute_tolerance'] = 1E-10
        prm['snes_solver']['relative_tolerance'] = 1E-6
        prm['snes_solver']['maximum_iterations'] = 1000
        prm['newton_solver']['absolute_tolerance'] = 1E-10
        prm['newton_solver']['relative_tolerance'] = 1E-6
        prm['newton_solver']['maximum_iterations'] = 1000
        prm['newton_solver']['relaxation_parameter'] = 1.0


        # print(dl.info(solver.parameters, True))
        iterations, converged = solver.solve()
        self.iterations += iterations
        state.zero()
        state.axpy(1., u.vector())


def HyperelasticityPrior(Vh_PARAMETER, pointwise_std, correlation_length, mean=None, anis_diff=None):
    # Delta and gamma
    delta = 1.0 / (pointwise_std * correlation_length)
    gamma = delta * correlation_length ** 2
    if anis_diff is None:
        theta0 = 1
        theta1 = 1
        alpha = math.pi / 4.
        anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree=1)
        anis_diff.set(theta0, theta1, alpha)
    if mean is None:
        return hp.BiLaplacianPrior(Vh_PARAMETER, gamma, delta, anis_diff, robin_bc=True)
    else:
        return hp.BiLaplacianPrior(Vh_PARAMETER, gamma, delta, anis_diff, mean=mean, robin_bc=True)


def HyperelasticityMisfit(prior, pde, Vu, settings):
    # Create misfit object
    ndim = 2
    # Uniform target
    ntargets = settings["ntargets"]
    aspect_ratio = settings["aspect_ratio"]
    # targets = np.random.uniform(0.05, 1-0.05, [ntargets, ndim])
    # targets[:, 0] = aspect_ratio*targets[:, 0]
    # if settings["verbose"]:
    #     print("Number of observation points: {0}".format(ntargets))
    # misfit = hp.PointwiseStateObservation(Vu, targets)
    rel_noise = settings.get("rel_noise")


    ny_targets = math.floor(math.sqrt(ntargets / aspect_ratio))
    nx_targets = math.floor(ntargets / ny_targets)
    if not ny_targets * nx_targets == ntargets:
        raise Exception("The number of obaservation points cannot lead to a regular grid \
        that is compatible with the aspect ratio.")
    targets_x = np.linspace(0.0, settings["aspect_ratio"], nx_targets + 2)
    targets_y = np.linspace(0.0, 1.0, ny_targets + 2)
    targets_xx, targets_yy = np.meshgrid(targets_x[1:-1], targets_y[1:-1])
    targets = np.zeros([ntargets, ndim])
    targets[:, 0] = targets_xx.flatten()
    targets[:, 1] = targets_yy.flatten()
    settings["targets"] = targets
    misfit = hp.PointwiseStateObservation(Vu, targets)

    utrue = pde.generate_state()
    if rel_noise is not None:
        noise_std_dev = 0.0
        print("Determining noise standard deviation choice based on 50 random samples")
        for i in range(50):
            utrue = pde.generate_state()
            mtrue = true_parameter(prior) #random true parameter
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

    return misfit, targets, noise_std_dev


def model(settings):
    np.random.seed(settings["seed"])
    output_path = settings["output_path"]

    if output_path is not None and not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)


    ndim = 2
    nx = settings["nx"]
    mesh = dl.RectangleMesh(dl.Point(0, 0), dl.Point(settings["aspect_ratio"], 1.0), \
                            nx, math.floor(nx / settings["aspect_ratio"]))
    Vh_STATE = dl.VectorFunctionSpace(mesh, 'Lagrange', 2)
    Vh_PARAMETER = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh_STATE, Vh_PARAMETER, Vh_STATE]

    # right_traction_expr = dl.Expression(("a*exp(-1.0*pow(x[1] - 0.5,2)/b)", "c*(1.0 + (x[1]/d))"), a=0.08, b=4, c=0.04,
    #                                     d=10,
    #                                     degree=5)
    # traction_func = dl.Function(Vh[STATE])
    # traction_func.interpolate(traction_expression)
    # right_traction = dl.Constant((0.0, 0.0))
    # pde = CustomHyperelasticityProblem(Vh, mesh, right_traction)
    pde = CustomHyperelasticityProblem(Vh, 1.5)

    theta0 = settings["theta0"]
    theta1 = settings["theta1"]
    alpha = settings["alpha"]
    anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree=5)
    anis_diff.set(theta0, theta1, alpha)
    prior = HyperelasticityPrior(Vh[PARAMETER], settings["sigma"], settings["rho"], anis_diff=anis_diff)

    misfit, targets, noise_stdev = HyperelasticityMisfit(prior, pde, Vh[STATE], settings)

    if settings["plot"] and output_path is not None: #This code is irrelevant because right now
        #we don't compute m true here. need to move plotting code to a separate place!
        obs_values = np.linalg.norm(misfit.d.get_local().reshape(settings["ntargets"], 2), axis=1)
        cbar = plt.scatter(targets[:, 0], targets[:, 1], c=obs_values, marker=",", s=10)
        plt.colorbar(cbar)
        plt.xticks([])
        plt.yticks([])
        plt.xlim([0, settings["aspect_ratio"]])
        plt.ylim([0, 1])
        # plt.gca().set_aspect(1./settings["aspect_ratio"])
        plt.gca().set_aspect('equal')
        plt.savefig(output_path + "obs_sample.pdf", bbox_inches="tight")
        plt.close()
        cbar = dl.plot(hp.vector2Function(mtrue, Vh[PARAMETER]))
        plt.colorbar(cbar)
        plt.axis("off")
        plt.savefig(output_path + "mtrue.pdf", bbox_inches="tight")
        plt.close()
        cbar = dl.plot(hp.vector2Function(pde.parameter_map(mtrue), Vh[PARAMETER]))
        plt.colorbar(cbar)
        plt.axis("off")
        plt.savefig(output_path + "Etrue.pdf", bbox_inches="tight")
        plt.close()
        cbar = dl.plot(hp.vector2Function(utrue, Vh[STATE]), mode="displacement")
        plt.colorbar(cbar)
        plt.axis("off")
        plt.savefig(output_path + "utrue.pdf", bbox_inches="tight")
        plt.close()
        np.save(output_path + "targets.npy", targets)
    rel_noise = settings.get("rel_noise")
    mtrue = None

    return pde, prior, misfit, mtrue, noise_stdev #mtrue kept only for legacy. Need to remove completely

def true_parameter(prior):
    noise = dl.Vector()
    prior.init_vector(noise, "noise")
    hp.parRandom.normal(1.0, noise)
    mtrue = dl.Vector()
    prior.init_vector(mtrue, 0)
    prior.sample(noise, mtrue)
    return mtrue
