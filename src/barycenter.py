"""
Stochastic Wasserstein Barycenter
"""

import PIL
from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
# import imageio
from scipy.spatial.distance import cdist


class SemiDiscreteOT:
    """

    """

    def __init__(self, sample_size: int = 64000, d: int = 2, **args):
        """
        Initialize.

        sample_size: sample size used in solving dual-variable
                     values via generalized Voronoi diagram
        d: dimension
        """
        self.d = 2
        self.sample_size = sample_size
        self.fig_samplers = {}
        self.samples = ()


    def simple_fig_sampler(self, figname:str, *args) -> tuple:
        """
        """
        if self.samples:
            return self.samples

        img = plt.imread(figname)
        match img.shape:
            case [n, m]:
                img = 1 - img
            case [n, m, 3]:
                grayscale_coef = np.array([0.299, 0.587, 0.114])
                img = 1 - np.einsum("ijk,k->ij", img, grayscale_coef)
        self.img = img
        positions = np.argwhere(img>0.)
        weights = img[img>0.]
        print(len(weights))
        self.samples = (positions, weights/weights.sum())
        return self.samples

    def figure_sampler(self,
                       figname: str,
                       sample_size=None,
                       channel: str = "L"):
        """
        channel: in "R", "G", "B" and "L", where "L" represents the grayscale
        """
        sample_size = (self.sample_size
                       if sample_size is None else sample_size)

        if (figname, channel) in self.fig_samplers:
            prob = self.fig_samplers[(figname, channel)]
        else:
            img = plt.imread(figname)  # May ues PIL library?
            MAX_V = 255 if np.max(img) > 1 else 1
            if len(img.shape) >= 3:
                grayscale_coef = np.array([0.299, 0.587, 0.114])
                if channel == "L":
                    img = MAX_V - np.einsum("ijk,k->ij", img[:, :, :3], grayscale_coef)
                elif channel == "R":
                    img = MAX_V - img[:, :, 0]
                elif channel == "G":
                    img = MAX_V - img[:, :, 1]
                elif channel == "B":
                    img = MAX_V - img[:, :, 2]
            else:
                    img = MAX_V - img

            prob = img / img.sum()
            self.fig_samplers[(figname, channel)] = prob

        n, m = prob.shape
        ret = np.random.choice(n * m, sample_size, p=prob.reshape(-1))
        return np.array([(n - x // m, x % m) for x in ret], dtype=float), np.ones(sample_size) / sample_size

    def iid_sampler(self, sampler: Callable, sample_size=None) -> tuple:
        """
        IID sampler
        """
        sample_size = (self.sample_size
                       if sample_size is None else sample_size)
        return np.array([sampler() for _ in range(sample_size)]), np.ones(sample_size) / sample_size



    def powercell_density_means(self,
                                disc: np.ndarray,
                                weights: np.ndarray,
                                samples: np.ndarray,
                                means: bool = True,
                                **kwargs) -> list[np.ndarray]:
        """
        disc: X, the discretization to test
        weights: generalized Voronoi diagram weights
        samples: samples from distributions
        kwargs["sample_wts"] if samples are not of equal weights.
        """
        # assert disc.shape[1] == self.d
        n = disc.shape[0]
        sample_size = samples.shape[0]
        sample_wts = kwargs.get("sample_wts", np.ones(sample_size))
        total_wts = np.sum(sample_wts)
        if n == 1:
            return np.ones(n)
        in_density = np.zeros(n)
        bary = np.zeros_like(disc)
        distances = cdist(disc, samples) - weights[:, np.newaxis]
        #                  (n, size)              (n, 1)
        idx = np.argmin(distances, axis=0)

        for i in range(sample_size):
            in_density[idx[i]] += sample_wts[i]
            if means:
                bary[idx[i], :] += samples[i, :] * sample_wts[i]

        bary[in_density > 0, :] /= in_density[in_density > 0].reshape(-1, 1)

        rho = in_density / total_wts  # sample_size

        return rho, (bary if means else None)

    def weight_update(self,
                      disc: np.ndarray,
                      weights: np.ndarray,
                      samples: np.ndarray,
                      max_step: int = 20000,
                      eps: float = 1e-3,
                      **kwargs):
        """
        Weight update
        """
        assert disc.shape[0] == weights.shape[
            0], f"{disc.shape} vs {weights.shape}"
        n = disc.shape[0]
        s_size = samples.shape[0]
        sample_wts = kwargs.get("sample_wts",
                                np.ones(s_size)/s_size)
        grad = 1 / n - self.powercell_density_means(disc,
                                                    weights,
                                                    samples,
                                                    False,
                                                    sample_wts=sample_wts)[0]

        alpha = 1e-3
        beta = 0.99
        z = np.zeros_like(weights)
        normGrad = np.linalg.norm(grad)
        iter = 1
        while normGrad > eps:
            if iter % 100 == 0:
                if iter % 1000 == 0:
                    print(f'Iter: {iter} (norm: {normGrad})')
                if iter > max_step:
                    break
            iter += 1

            z = beta * z + grad
            weights += alpha * z

            grad = 1 / n - self.powercell_density_means(
                disc, weights, samples, False, sample_wts=sample_wts)[0]
            normGrad = np.linalg.norm(grad)

        return weights

    # Function to update powercell

    def powercell_update(self,
                         disc: np.ndarray,
                         weights: np.ndarray,
                         samplers: list[Callable],
                         initial_sampler: Callable,
                         powercell_step: int = 1,
                         sample_size=None,
                         max_step=20000, **kwargs):
        """
        X: [n, d] array
        w: [n, m] array
        mu: (m)-many samplers to take barycenter.
        """

        sample_size = self.sample_size if sample_size is None else sample_size
        n = disc.shape[0] + 1
        m = len(samplers)

        # Sample new point and update weights
        y = initial_sampler(1)
        if n == 1:
            X = np.array([y])
            w = np.zeros((1, m))
        else:
            # print("===", disc, y, weights, np.zeros((1, m)))
            X = disc
            w = weights
        n = X.shape[0]
        # Run this for T iterations to ensure convergence (T = 10 works)
        for t in range(powercell_step):
            print("Update Round:",t)
            # Update weights
            samples = [None] * m
            sample_wts = [None] * m
            for k in range(m):
                samples[k], sample_wts[k] = samplers[k](sample_size)
                w[:, k] = self.weight_update(X,
                                             w[:, k],
                                             samples[k],
                                             max_step=max_step,
                                             sample_wts=sample_wts[k])

            # Move points
            # n = X.shape[0]
            Xnew = np.zeros_like(X)
            Msum = np.zeros(n)
            for k in range(m):
                M, B = self.powercell_density_means(X, w[:, k],
                                                    samples[k],
                                                    True,
                                                    sample_wts=sample_wts[k])

                print(M.shape, B.shape)
                Xnew += np.einsum("i,ij->ij", M, B)
                Msum += M

            for i in range(n):
                if Msum[i] > 0:
                    X[i, :] = Xnew[i, :] / Msum[i]

        return X, w

    def discretize_img_step(self, disc, weights, figname: str,
                            discrete_size: int, **args):
        """
        Discretize a single image
        """
        img = plt.imread(figname)
        n, m = img.shape[:2]
        samplers = [lambda k: self.simple_fig_sampler(figname, k)]
        init_sampler = lambda k: self.iid_sampler(
            lambda: (np.array([n, m]) * np.random.rand(2)), k)

        disc = init_sampler(discrete_size)[0] if disc is None else disc
        weights = np.random.rand(discrete_size,
                                 1) if weights is None else weights
        return self.powercell_update(disc, weights, samplers, init_sampler,
                                     **args)


def main():
    """
    A Test
    """
    sd = SemiDiscreteOT(5000)
    X = None
    w = None
    repeats = 30
    log = []

    for i in range(repeats):
        print("\n", "=" * 80 + "\n", i)
        X, w = sd.discretize_img_step(X, w, "../test.png", 10, max_step=20000)
        log = [[X, w]]

        import json
        with open(f"test_log_{i}.json", "w") as fp:
            fp.write(
                json.dumps([X.tolist(), w.tolist()],
                           indent=4))


if __name__ == '__main__':
    main()
