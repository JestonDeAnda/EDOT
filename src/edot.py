import numpy as np
import multiprocessing as mp
import scipy as sc
import itertools
import pickle
from scipy import stats as st
"""
"""

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from src import sinkhorn_numpy as sk
from . import EDOT_discretization as ed

import json

from .barycenter import *


class EDOT(SemiDiscreteOT):


    def fig_sampler(self, fig:str | np.ndarray, sample_size):
        match fig:
            case str(f):
                img = plt.imread(f)
            case None:
                img = self.img
            case _:
                img = fig

        prob = img / np.sum(img)
        n, m = img.shape[:2]
        ret = np.random.choice(n*m, sample_size, p=prob.reshape(-1))
        return np.array([(n - x // m, x % m) for x in ret], dtype=float)


    def lagrange_descretize(self, disc, positions, weights,
                            repeats=500,
                            alpha=0.05, beta=0.95, zeta=0.004,
                            **kwargs):
        """
        disc and positions should be inside `[0, 1] * [0, 1]` region.
        """
        X = disc
        W = np.ones(X.shape[0]) / X.shape[0]
        z = 0
        eps = 1e-5
        for i in range(repeats):
            grad = ed.gradient_POS(positions, weights,
                                   X, W,
                                   zeta=zeta)
            z = beta * z + grad[0]
            X -= alpha * z

            if i%50 == 0:
                print(i, (L:=np.linalg.norm(grad[0])))
                if L < eps:
                    return disc

        return disc

    def discretize_img_step(self, disc, weights,
                            figname:str, discrete_size:int, **args):
        """
        """
        pos, wts = self.simple_fig_sampler(figname)
        n, m = self.img.shape[:2]
        pos = np.asarray(pos, dtype=float) / max(n, m)
        disc = self.fig_sampler(self.img, discrete_size) if disc is None else disc
        disc = np.asarray(disc, dtype=float) / max(n, m)
        disc += np.random.rand(*disc.shape)*1e-2
        print(disc)
        disc = self.lagrange_descretize(disc, pos, wts, **args)
        return disc*max(n, m), weights

def main():
    """
    Test EDOT
    """
    figname = "./test.png"
    sd = SemiDiscreteOT(5000)
    img = plt.imread(figname)
    n, m = img.shape[:2]
    sampler = lambda k: sd.figure_sampler(figname, k)
    init_sampler = lambda k: sd.iid_sampler(
            lambda: (np.array([n, m]) * np.random.rand(2)), k)

    alpha = 0.05
    beta = 0.95
    tgt_sz = 10
    sample_sz = 1000
    X = init_sampler(tgt_sz)/n

    z = np.zeros_like(X)
    repeats = 1000
    for i in range(repeats):

        sample = sampler(sample_sz)/n
        # print(X, sample)
        grad = ed.gradient_POS(sample, np.ones(sample_sz)/sample_sz,
                               X, np.ones(tgt_sz)/tgt_sz, zeta=0.005)
        z = beta * z + grad[0]
        X -= alpha * z

        if i%50 == 0:
            print(i, np.linalg.norm(grad[0]))
            sample_sz += 20
            with open(f"edot_test_{i}.json", "w") as fp:
                fp.write(
                    json.dumps({"edot_lagrange":X.tolist(), "scale":n},
                               indent=4))



if __name__ == '__main__':
    main()
