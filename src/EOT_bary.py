import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

import json

# from src import sinkhorn_numpy as sk
from . import EDOT_discretization as ed

from .sinkhorn_numpy import *
from .barycenter import *
from .edot import *


class EOTBary(EDOT):

    @staticmethod
    def find_T(C: np.ndarray, M: np.ndarray, eps: float = 0.2):
        C = C.copy()
        n, _ = C.shape
        M[M < eps] = 0

        dw = {}
        tree = {i: {i} for i in range(n)}

        for j, v in enumerate(M.T):
            u = np.argwhere(v > 0).reshape(-1)
            for ii, s in enumerate(u):
                tree_s = tree[s]
                for t in u[ii + 1:]:
                    tree_t = tree[t]
                    if tree_s != tree_t:
                        tree_s.update(tree_t)
                        for idx in tree_t:
                            tree[idx] = tree_s
                    s1, t1 = (s, t) if s < t else (t, s)
                    if (s1, t1) not in dw:
                        dw[(s1, t1)] = [0, 0]  # [sum, count]
                    dw[(s1, t1)][0] += C[s1, j] - C[t, j]
                    dw[(s1, t1)][1] += 1

        for key in dw:
            dw[key] = dw[key][0] / dw[key][1]

        x = np.zeros(n)
        while len(set(map(frozenset, tree.values()))) > 1:
            
            key, val = dw.popitem()
            keys = {key[0], key[1]}
            x[key[1]] = x[key[0]] + val
            while dw:
                breadth = [k for k in dw.keys() if keys.intersection(k)]
                for key in breadth:
                    if key[0] in keys and key[1] in keys:
                        dw.pop(key)
                    elif key[0] in keys:
                        x[key[1]] = x[key[0]] + dw.pop(key)
                    else:
                        x[key[0]] = x[key[1]] - dw.pop(key)
                    keys.update(key)
            return -x
        
        return np.zeros_like(x)

    @staticmethod
    def find_T_old(C, M, eps=2e-1):
        """
        find w given distance C and M.
        y_i\in V_i if x_i = argmin_{i} C[i,j] - w[i]
        """

        # C = np.round(C, 4)
        # print(C.T, M.T)
        C = C.copy()
        n, m = C.shape
        T = np.zeros(n)
        M[M < eps] = 0
        # for ii, v in enumerate(np.round(M.T, 2)):
        #     print(ii, v)

        dw = {}
        tree = [{i} for i in range(n)]
        for j, v in enumerate(M.T):
            u = np.argwhere(v > 0).reshape(-1)
            for ii, s in enumerate(u):
                tree_s = [x for x in tree if s in x]

                for t in u[ii + 1:]:
                    tree_t = [x for x in tree if t in x]
                    if len(tree_t) == 0:
                        break
                    # print(tree_t, tree_u)
                    # if tree_t[0] != tree_s[0]:
                    tree_s[0].update(tree_t[0])
                    if tree_s != tree_t:
                        tree.remove(tree_t[0])
                    s1, t1 = (s, t) if s < t else (t, s)
                    dw[(s1, t1)] = dw.get((s1, t1), []) + [
                        C[s1, j] - C[t, j],
                    ]
                    # print(dw[(u[ii], t)], tree_t[0], tree_s[0])

        # print([(key, np.mean(val), np.std(val)) for key, val in dw.items()])
        # print(tree)
        for key, val in dw:
            val = np.mean(val)
        x = np.zeros(n)
        if len(tree) == 1:
            key, val = dw.popitem()
            keys = {key[0], key[1]}
            x[key[1]] = x[key[0]] + np.mean(val)
            count = 0
            while len(dw) > 0 and (count := count + 1) < 10:
                # print(list([keys.intersection(k), k] for k in dw.keys()))
                breadth = [k for k in dw.keys() if keys.intersection(k)]
                # print(breadth, keys)
                for key in breadth:
                    if key[0] in keys and key[1] in keys:
                        dw.pop(key)
                    elif key[0] in keys:
                        x[key[1]] = x[key[0]] + np.mean(dw.pop(key))
                    else:
                        x[key[0]] = x[key[1]] - np.mean(dw.pop(key))
                    keys.update(key)
            # print(x)
            return np.zeros_like(x)

        aux = -np.ones([n, n]) * np.inf
        for i, j in np.argwhere(M > 0):
            C[:, j] -= C[i, j]
            # print(i, j)
            aux[i] = np.maximum(C[:, j], aux[i])

        x = np.zeros(n)
        x[1:] = aux[1:, 0] - aux[0, 0]
        # print(x)
        for i in range(10):
            D = aux[1:, 1:] - np.diag(aux[1:, 1:]).reshape(-1, 1)
            y = np.minimum(x[1:], np.min(D, axis=0))
            print("Round", i, x)
            if np.sum(np.abs(y - x[1:])) < 1e-3:
                break
            x[1:] = y

        return x

    def weight_update(self,
                      disc: np.ndarray,
                      weights: np.ndarray,
                      samples: np.ndarray,
                      max_step: int = 1000,
                      eps: float = 0.01,
                      **kwargs):
        """
        Weight update
        """
        assert disc.shape[0] == weights.shape[
            0], f"{disc.shape} vs {weights.shape}"
        n = disc.shape[0]
        s_size = samples.shape[0]
        sample_wts = kwargs.get("sample_wts", np.ones(s_size) / s_size)

        grad = 1 / n - self.powercell_density_means(
            disc, weights, samples, False, sample_wts=sample_wts)[0]

        z = np.zeros_like(weights)
        normGrad = np.linalg.norm(grad)
        C = cdist(disc, samples) / kwargs.get("data_upper", 64)
        if normGrad > 0.05:
            # print(f"===========\nBefore Sinkhorn: (norm: {normGrad})\n", weights)
            M = sinkhorn_numpy(np.exp(-C / 0.01), np.ones(n) / n, sample_wts)
            M = M / np.sum(M, axis=0)
            weights = self.find_T(C, M) * kwargs.get("data_upper", 64)
        print(f"{n}, {s_size}\nAfter Sinkhorn:\n", weights)
        grad = 1 / n - self.powercell_density_means(
            disc, weights, samples, False, sample_wts=sample_wts)[0]

        alpha = 1e-2
        beta = 0.99
        z = np.zeros_like(weights)
        normGrad = np.linalg.norm(grad)
        iter = 0
        while normGrad > eps:
            if iter % 50 == 0:
                print(f'Iter: {iter} (norm: {normGrad}), {n}, {s_size}')
                if iter > max_step:
                    break
            iter += 1

            z = beta * z + grad
            weights += alpha * z

            grad = 1 / n - self.powercell_density_means(
                disc, weights, samples, False, sample_wts=sample_wts)[0]
            normGrad = np.linalg.norm(grad)

        #### Extra
        dist = C * 64 - weights[:, np.newaxis]
        idx = np.argmin(dist, axis=0)

        print("END:\n", weights - weights[0])
        return weights

    def discretize_img_step(self, disc, weights, figname: str,
                            discrete_size: int, **args):
        """
        Combined.
        """
        return super(EDOT, self).discretize_img_step(disc, weights, figname,
                                                     discrete_size, **args)


def main_edot():
    """
    Test EDOT
    """
    figname = "./test.png"
    sd = SemiDiscreteOT(5000)
    img = plt.imread(figname)
    n, m = img.shape[:2]
    sampler = lambda k: sd.simple_fig_sampler(figname, k)
    init_sampler = lambda k: sd.iid_sampler(
        lambda: (np.array([n, m]) * np.random.rand(2)), k)

    alpha = 0.05
    beta = 0.95
    tgt_sz = 10
    sample_sz = 1000
    X = init_sampler(tgt_sz)[0] / n
    z = np.zeros_like(X)
    repeats = 1000
    for i in range(repeats):

        sample = sampler(sample_sz)[0] / n
        # print(X, sample)
        grad = ed.gradient_POS(sample,
                               np.ones(sample_sz) / sample_sz,
                               X,
                               np.ones(tgt_sz) / tgt_sz,
                               zeta=0.005)
        z = beta * z + grad[0]
        X -= alpha * z

        if i % 100 == 0:
            print(i, np.linalg.norm(grad[0]))
            sample_sz += 20
            with open(f"edot_test_{i}.json", "w") as fp:
                fp.write(
                    json.dumps({
                        "edot_lagrange": X.tolist(),
                        "scale": n
                    },
                               indent=4))


def main():
    """
    A Test
    """
    sd = EOTBary(5000)
    X = None
    w = None
    repeats = 1
    log = []

    for i in range(repeats):
        print("\n", "=" * 80 + "\n", i)
        X, w = sd.discretize_img_step(X,
                                      w,
                                      "./test.png",
                                      10,
                                      powercell_step=11,
                                      max_step=400)
        log = [[X, w]]

        import json
        with open(f"test_log_{i}.json", "w") as fp:
            fp.write(json.dumps([X.tolist(), w.tolist()], indent=4))


if __name__ == '__main__':
    # main()

    # 示例数据
    C = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    M = np.array([[1.0, 0.5, 0.2], [0.9, 1.0, 0.1], [0.8, 0.7, 1.0]])

    T = find_T(C, M)
    print(T)
