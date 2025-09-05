"""Calculate Wasserstein in Batch: application"""

import torch
from .wasserstein_gpu import *


DEVICE = "cuda"

BINARY_THRES = 0.5


def wasserstein_sample_fig(sample:torch.Tensor, fig:torch.Tensor)->torch.Tensor:
    """
    Parameters
    ----------
    sample: torch.Tensor, shape (K * m * d), K is batch size, m is object amount, d is dimension of space
            sample is a batch of targets generated from score-matching algorithm
    fig: torch.Tensor, processed figure, cut and rescaled to its bounding box. 
         [NOTE] Must be of shape 64*64 !!!

    Return
    ------
    Wasserstein distance. should be of shape [K]
    """
    fig_pos = (torch.argwhere(fig>BINARY_THRES) - 31.5) / 32
    return wasserstein_distance_batch(sample, None, fig_pos, None)
