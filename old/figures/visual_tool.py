import logging
import math
import os
import sys

import dolfin as dl
import hippylib as hp
import matplotlib.pyplot as plt
import numpy as np
import ufl

plt.rc("text", usetex=True)
plt.rc("font", family="serif", size=20)
import matplotlib

matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

STATE, PARAMETER, ADJOINT = 0, 1, 2


class InternalDomain(dl.SubDomain):
    def __init__(self, box):
        self.box = box

    def inside(self, x, on_boundary):
        return (
            (x[0] > self.box[0] - dl.DOLFIN_EPS)
            and (x[1] > self.box[1] - dl.DOLFIN_EPS)
            and (x[2] < self.box[2] + dl.DOLFIN_EPS)
            and (x[2] < self.box[2] + dl.DOLFIN_EPS)
        )


def plot_component(Vh, vec, component=0, box=None):
    u_sub = hp.vector2Function(vec, Vh).split()[component]
    if not box is None:
        domains = dl.CellFunction("size_t", Vh.mesh(), Vh.mesh().topology().dim())
        subdomain = InternalDomain(box)
        domains.set_all(0)
        subdomain.mark(domains, 1)
        dx = dl.Measure("dx")(subdomain_data=domains)
    cbar = dl.plot(u_sub)
    plt.axis("off")
    plt.colorbar(cbar)
