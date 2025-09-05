

import torch
from typing import List, Tuple, Callable
from matplotlib import pyplot as plt
from torch.optim import Adam, SGD



import numpy as np

DEVICE = "cuda"

GRID = None
ARANGE = torch.arange(500000, device=DEVICE)

def make_grid(shape):
    return torch.stack(torch.meshgrid(torch.arange(shape[0], device=DEVICE),
                                      torch.arange(shape[1], device=DEVICE),
                                      indexing='ij'),
                       dim=-1)
def cdist(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    x1_exp = x1.unsqueeze(1)
    x2_exp = x2.unsqueeze(0)
    dist = torch.sum((x1_exp - x2_exp) ** 2, dim=2)
    dist = torch.sqrt(dist)
    return dist

class SemiDiscreteOTTorch:



    def __init__(self, sample_size: int = 4000, d: int = 2, **args):
        """
        Initialize
        """
        self.d = 2
        self.sample_size= sample_size
        self.fig_samplers = {}
        self.samples = ()
        self.optimizer = None


    def simple_fig_sampler(self, figname: str, *args) -> tuple:
        """

        Return
        ------
        tuple (positions, weights)
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

        img = torch.tensor(img, device=DEVICE, dtype=torch.float32)
        return self.cuda_fig_sampler(img)


    def cuda_fig_sampler(self, img:torch.Tensor) -> tuple:
        """
        """
        print("MAKE_GRID")
        grid = make_grid(img.shape)
        positions = (img>0.5).nonzero(as_tuple=True)[0]
        self.samples = (grid[positions].reshape(-1,2), img[positions].reshape(-1))
        print(self.samples)
        return self.samples



    def iid_sampler(self, sampler: Callable, sample_size=None):
        """
        IID sampler
        """
        sample_size = (self.sample_size
                       if sample_size is None else sample_size)
        return torch.tensor([sampler() for _ in range(sample_size)],device=DEVICE)

        

    def powercell_density_means(self,
                            disc: torch.Tensor,
                            weights: torch.Tensor,
                            samples: torch.Tensor,
                            means: bool = True,
                            **kwargs) -> list[torch.Tensor]:
        n = disc.shape[0]
        sample_size = samples.shape[0]
        sample_wts = kwargs.get("sample_wts",
                                torch.ones(sample_size)).float()  # shape=[sample_size]
        weighted_samples = kwargs.get("weighted_samples",
                                      samples*sample_wts.reshape(-1,1)) # shape=[sample_size, 2]

        total_wts = torch.sum(sample_wts)
        if n == 1:
            return torch.ones(n, device=DEVICE)
        in_density = torch.zeros(n, device=DEVICE)
        bary = torch.zeros_like(disc, device=DEVICE) # shape = [n, 2]
        distances = cdist(disc, samples) - weights[:, None]
        idx = torch.argmin(distances, axis=0)
        aux = torch.zeros_like(distances,
                               device=DEVICE,
                               dtype=torch.float32)  # shape = [n, sample_size]
        aux[idx, ARANGE[:sample_size]] = 1
        in_density = torch.einsum("ij,j->i", aux, sample_wts)

        if means:
            bary = torch.einsum("ij,jk->ik", aux, weighted_samples)

        # for i in range(sample_size):
        #     in_density[idx[i]] += sample_wts[i]
        #     if means:
        #         bary[idx[i], :] += samples[i, :] * sample_wts[i]

        bary[in_density > 0, :] /= in_density[in_density > 0].reshape(-1, 1)

        rho = in_density / total_wts
        return rho, (bary if means else None)

    def weight_update(self,
                      disc: torch.Tensor,
                      weights: torch.Tensor,
                      samples: torch.Tensor,
                      max_step: int = 20000,
                      eps: float = 1e-3,
                      **kwargs):
        """
        Weight update
        """
        assert disc.shape[0] == weights.shape[0], f"{disc.shape} vs {weights.shape}"
        n = disc.shape[0]
        s_size = samples.shape[0]
        sample_wts = kwargs.pop("sample_wts", torch.ones(s_size) / s_size)

        print("TYPE", weights.dtype)
        # optimizer = kwargs.get("optimizer")
        optimizer = SGD([weights], lr=0.001, momentum=0.99)
        optimizer.zero_grad()
        weights.grad = self.powercell_density_means(disc,
                                                    weights,
                                                    samples,
                                                    False,
                                                    sample_wts=sample_wts,
                                                    **kwargs)[0] - 1 / n

        alpha = 1e-3
        beta = 0.99
        
        # z = torch.zeros_like(weights, device=DEVICE)
        normGrad = torch.norm(weights.grad)
        index = 1
        while normGrad > eps:
            # print("wt-update", iter)
            if index % 200 == 0:
                print(f'Iter: {index} (norm: {normGrad.item()})')
                if index > max_step:
                    break
            index += 1
            optimizer.step()
            optimizer.zero_grad()

            weights.grad = self.powercell_density_means(disc,
                                                        weights,
                                                        samples,
                                                        False,
                                                        sample_wts=sample_wts,
                                                        **kwargs)[0] - 1 / n
            normGrad = torch.norm(weights.grad)

        return None


    def powercell_update(self,
                         disc: torch.Tensor,
                         weights: torch.Tensor,
                         samplers: list[Callable],
                         initial_sampler: Callable,
                         powercell_step: int = 15,
                         sample_size=None,
                         max_step=20000, **kwargs):
        """
        X: [n, d] tensor
        w: [n, m] tensor
        mu: (m)-many samplers to take barycenter.
        """

        sample_size = self.sample_size if sample_size is None else sample_size
        n = disc.shape[0] + 1
        m = len(samplers)
        
        # Sample new point and update weights
        y = initial_sampler(1)
        if n == 1:
            X = torch.tensor([y], device=DEVICE)
            w = torch.zeros((1, m), device=DEVICE)
        else:
            X = disc
            w = [x.clone().detach().requires_grad_() for x in weights]
        self.optimizer = [SGD([x], lr=0.001, momentum=0.99) for x in w]
        n = X.shape[0]
        Xnew = torch.zeros_like(X, device=DEVICE)
        Msum = torch.zeros(n, device=DEVICE)
        samples = [None] * m
        sample_wts = [None] * m
        weighted_samples = [None] * m
        # Run this for T iterations to ensure convergence (T = 10 works)
        for t in range(powercell_step):
            # Update weights
            for k in range(m):
                samples[k], sample_wts[k] = samplers[k](sample_size)
                weighted_samples[k] = samples[k] * sample_wts[k].reshape(-1,1)
                print("sample_get", k)
                # samples[k] = samples[k].clone().detach()
                # sample_wts[k] = sample_wts[k].clone().detach()
                self.weight_update(X, w[k],
                                   samples[k],
                                   max_step=max_step,
                                   sample_wts=sample_wts[k],
                                   weighted_samples=weighted_samples[k],
                                   optimizer=self.optimizer[k])

            # Move points
            Xnew *= 0
            Msum *= 0
            
            for k in range(m):
                M, B = self.powercell_density_means(X, w[k], samples[k],
                                                    True,
                                                    sample_wts=sample_wts[k],
                                                    weighted_samples=weighted_samples[k])

                print(']]]]]]]]]]]]', M.shape, B.shape)
                Xnew += M.unsqueeze(1) * B
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

        if disc is None:
            disc = init_sampler(discrete_size)
        if weights is None:
            weights = [np.random.rand(discrete_size)]
            weights = [torch.from_numpy(x).float().to(DEVICE) for x in weights]
            


        return self.powercell_update(disc, weights, samplers, init_sampler,
                                     **args)



def main():
    """
    A Test
    """
    sd = SemiDiscreteOTTorch(5000)
    X = None
    w = None
    repeats = 30
    log = []

    for i in range(repeats):
        print("\n", "=" * 80 + "\n", i)
        X, w = sd.discretize_img_step(X, w, "../test.png", 10, max_step=2000)
        log = [[X, w]]

        import json
        with open(f"test_log_{i}.json", "w") as fp:
            fp.write(
                json.dumps([X.tolist(), w[0].tolist()],
                           indent=4))


if __name__ == '__main__':
    main()
    exit()
    from barycenter import SemiDiscreteOT
    sdt = SemiDiscreteOTTorch(5000)
    sd = SemiDiscreteOT(5000)
    X = None
    w = None
    repeats = 30
    log = []


    fig = sd.figure_sampler("../test.png")
    figt = sdt.simple_fig_sampler("../test.png")
    print(fig, figt)
    fign = figt.cpu().numpy()
    print(fig - fign)
