#!/usr/bin/env python
'''
EDOT_disc_torch

PyTorch implementation of EDOT discretization functions.
Requires `torch` and `sinkhorn_torch`.

Date: 2025.09.05
'''

import torch
from . import sinkhorn_torch as sk


def uniform_sampler(size):
    '''
    Generate uniform samples in [0, 1] with uniform weights.
    
    Parameters
    ----------
    size: int, number of samples
    
    Return
    ------
    (positions, weights): tuple
        positions: torch tensor of shape (size, 1) with values in [0, 1]
        weights: torch tensor of shape (size) with uniform weights
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    positions = (torch.linspace(0, 1 - (1. / size), size,
                                device=device).reshape(-1, 1) + (0.5 / size))
    weights = torch.ones(size, dtype=torch.float, device=device) / size
    return positions, weights


def construct_transport_matrix(n, m, k, zeta, sample, target, exp, distance,
                               distance_gradient):
    '''
    Construct transport matrix for EOT calculations.
    
    Parameters
    ----------
    n: int, number of samples
    m: int, number of targets
    k: float, parameter in k-Wasserstein distance W_k^k
    zeta: float, regularization parameter in EOT
    sample: torch tensor of shape (n, d), sample points
    target: torch tensor of shape (m, d), target points
    exp: float, exponent for distance calculation
    distance: callable, function to calculate distance matrix
    distance_gradient: callable, function to calculate gradient of distance
    
    Return
    ------
    (dist, dist_grad, M): tuple
        dist: torch tensor of shape (n, m), distance matrix
        dist_grad: torch tensor of shape (n, m, d), gradient of distance
        M: torch tensor of shape (n, m), transport matrix
    '''
    # Calculate distance matrix
    dist = distance(sample, target, exp)
    # Calculate gradient of distance
    dist_grad = distance_gradient(sample, target, exp)

    # Adjust gradient if exp != k
    if exp != k:
        dist_grad *= ((k / exp) * (dist**(k / exp - 1))).reshape(n, m, 1)
        LM = -(dist)**(k / exp) / zeta
    else:
        LM = -dist / zeta

    # Stabilize by subtracting maximum value
    LM -= torch.max(LM)
    # Calculate transport matrix
    M = torch.exp(LM)

    return dist, dist_grad, M


def iid_target_sampler_1d_uniform(target_size):
    '''
    Sample the iid samples from 1d uniform distribution
    
    Parameters
    ----------
    target_size: int, number of samples
    
    Return
    ------
    (position, weights): tuple
        position: torch tensor of shape (target_size, 1) with values in [0,1]
        weights: torch tensor of shape (target_size) of sum 1
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_pos = torch.rand(target_size, 1, device=device)
    # For Dirichlet distribution equivalent, use concentration parameter 10
    concentration = torch.ones(target_size, device=device) * 10.0
    target_weight = torch.distributions.Dirichlet(concentration).sample()
    return target_pos, target_weight


def euclidean_distance(start, end, exp=2.):
    '''
    Euclidean distance between two sets of points
    
    Parameters
    ----------
    start: torch tensor of shape (n, d), n starting points
    end: torch tensor of shape (m, d), m ending points
    exp: float, exponent for distance calculation
    
    Return
    ------
    distance matrix of shape (n, m)
    '''
    # Ensure inputs are at least 2D
    if len(start.shape) == 1:
        start = start.reshape(-1, 1)
    if len(end.shape) == 1:
        end = end.reshape(-1, 1)

    # Get dimensions
    n, d = start.shape
    m = end.shape[0]

    # Calculate pairwise distances
    # Reshape for broadcasting: (n, 1, d) - (1, m, d) -> (n, m, d)
    diff = start.reshape(n, 1, d) - end.reshape(1, m, d)
    # Take absolute value, raise to power, and sum across dimensions
    dist = torch.sum(torch.abs(diff)**exp, dim=2)

    return dist


def euc_dist_grad(start, end, exp=2.):
    '''
    Gradient of the exp-distance between two sets of points with respect to end points
    
    Parameters
    ----------
    start: torch tensor of shape (n, d), n starting points
    end: torch tensor of shape (m, d), m ending points
    exp: float, exponent for distance calculation
    
    Return
    ------
    gradient tensor of shape (n, m, d)
    '''
    # Ensure inputs are at least 2D
    if len(start.shape) == 1:
        start = start.reshape(-1, 1)
    if len(end.shape) == 1:
        end = end.reshape(-1, 1)

    # Get dimensions
    n, d = start.shape
    m = end.shape[0]

    # Calculate difference tensor
    tmp = start.reshape(n, 1, d) - end.reshape(1, m, d)

    # Calculate gradient
    # |x|^(exp-1) * sign(x) * exp
    ret = torch.abs(tmp)**(exp - 1)
    # Apply sign: negative where tmp > 0, positive where tmp < 0
    ret = torch.where(tmp > 0, -ret, ret)

    return exp * ret


def wasserstein_distance(sample,
                         sample_weight,
                         target,
                         target_weight,
                         k=2.,
                         exp=2.,
                         zeta=0.1,
                         distance=euclidean_distance,
                         distance_gradient=euc_dist_grad):
    '''
    Calculate Wasserstein distance using Sinkhorn algorithm
    
    Parameters
    ----------
    sample: torch tensor of shape (sample_size, space_dimension)
    sample_weight: torch tensor of shape (sample_size)
    target: torch tensor of shape (target_size, space_dimension)
    target_weight: torch tensor of shape (target_size)
    k: float, parameter in k-Wasserstein distance W_k^k
    exp: float, exponent for distance calculation
    zeta: float, regularization parameter in EOT
    distance: callable, function to calculate distance matrix
    distance_gradient: callable, function to calculate gradient of distance
    
    Return
    ------
    float, Wasserstein distance
    '''
    # Ensure inputs are at least 2D
    if len(sample.shape) == 1:
        sample = sample.reshape(-1, 1)
    if len(target.shape) == 1:
        target = target.reshape(-1, 1)

    # Get dimensions
    n, d = sample.shape
    m = target.shape[0]

    # Construct transport matrix
    dist, dist_grad, M = construct_transport_matrix(n, m, k, zeta, sample,
                                                    target, exp, distance,
                                                    distance_gradient)

    # Calculate cost matrix
    cost = dist if exp == k else dist**(k / exp)

    # Apply Sinkhorn algorithm
    B_0 = sk.sinkhorn_torch(M.clone(),
                            row_sum=sample_weight,
                            col_sum=target_weight)

    # Calculate Wasserstein distance
    return torch.sum(B_0 * cost)


def cont_wasserstein(sampler,
                     target,
                     target_weight,
                     N=400,
                     ratio=2,
                     distance=wasserstein_distance,
                     zeta=0.01,
                     k=2.,
                     exp=2.):
    '''
    Calculate Wasserstein distance of continuous distributions using Richardson Extrapolation
    
    Parameters
    ----------
    sampler: callable, function that generates samples
    target: torch tensor of shape (m, d)
    target_weight: torch tensor of shape (m)
    N: int, sample size in Richardson extrapolation
    ratio: int, ratio of sample sizes used in Richardson extrapolation
    distance: callable, function to calculate Wasserstein distance
    zeta: float, regularization parameter in EOT
    k: float, parameter in k-Wasserstein distance W_k^k
    exp: float, exponent for distance calculation
    
    Return
    ------
    (W_inf, [W_1, W_2]): tuple
        W_inf: float, extrapolated Wasserstein distance
        [W_1, W_2]: list, Wasserstein distances at different sample sizes
    '''
    # Generate samples with size N
    sample, sample_weight = sampler(N)

    # Calculate Wasserstein distance with N samples
    W_1 = distance(sample,
                   sample_weight,
                   target,
                   target_weight,
                   zeta=zeta,
                   k=k,
                   exp=exp)

    # Generate samples with size N*ratio
    sample, sample_weight = sampler(N * ratio)

    # Calculate Wasserstein distance with N*ratio samples
    W_2 = distance(sample,
                   sample_weight,
                   target,
                   target_weight,
                   zeta=zeta,
                   k=k,
                   exp=exp)

    # Apply Richardson extrapolation
    t = k / target.shape[1]  # Dimension-dependent factor
    W_inf = (W_2 * ratio**t - W_1) / (ratio**t - 1)

    return W_inf, [W_1, W_2]


def gradient_EOT(sample,
                 sample_weight,
                 target,
                 target_weight,
                 k=2.,
                 exp=2.,
                 zeta=0.1,
                 distance=euclidean_distance,
                 distance_gradient=euc_dist_grad):
    '''
    Calculate gradient of the EOT total cost
    
    Parameters
    ----------
    sample: torch tensor of shape (sample_size, space_dimension)
    sample_weight: torch tensor of shape (sample_size)
    target: torch tensor of shape (target_size, space_dimension)
    target_weight: torch tensor of shape (target_size)
    k: float, parameter in k-Wasserstein distance W_k^k
    exp: float, exponent for distance calculation
    zeta: float, regularization parameter in EOT
    distance: callable, function to calculate distance matrix
    distance_gradient: callable, function to calculate gradient of distance
    
    Return
    ------
    (D_nu_obj, D_y_obj, total_cost): tuple
        D_nu_obj: torch tensor of shape (m), gradient with respect to target weights
        D_y_obj: torch tensor of shape (m, d), gradient with respect to target positions
        total_cost: float, total EOT cost
    '''
    # Ensure inputs are at least 2D
    if len(sample.shape) == 1:
        sample = sample.reshape(-1, 1)
    if len(target.shape) == 1:
        target = target.reshape(-1, 1)

    # Get dimensions
    n, d = sample.shape
    m = target.shape[0]

    # Construct transport matrix
    dist, dist_grad, M = construct_transport_matrix(n, m, k, zeta, sample,
                                                    target, exp, distance,
                                                    distance_gradient)

    # Apply Sinkhorn algorithm
    B_0 = sk.sinkhorn_torch(M.clone(),
                            row_sum=sample_weight,
                            col_sum=target_weight)
    B = B_0[:, :-1]  # All columns except the last
    A = sample_weight.clone()  # Diagonal matrix of sample weights
    D = target_weight[:-1].clone()  # All target weights except the last

    # Construction of Sinkhorn Information Matrix (SIM)
    SIM = torch.zeros((n + m - 1, n + m - 1),
                      dtype=torch.float,
                      device=sample.device)

    # Calculate components of SIM
    C = B.t() / A  # Transpose of B divided by A
    E = torch.diag(D) - torch.matmul(C, B)  # Diagonal matrix of D minus C*B
    E = torch.inverse(E)  # Inverse of E

    # Fill SIM blocks
    SIM[n:, n:] = E
    SIM[n:, :n] = -torch.matmul(E, C)
    SIM[:n, n:] = SIM[n:, :n].t()
    SIM[:n, :n] = torch.diag(1. / A) + torch.matmul(torch.matmul(C.t(), E), C)

    # Scale SIM by zeta
    SIM = SIM * zeta

    # Calculate derivatives on weights and positions
    D_nu_N = torch.zeros((n + m - 1, m),
                         dtype=torch.float,
                         device=sample.device)
    D_nu_N[:n, :] = -B_0 / target_weight.reshape(1, -1)

    # Calculate derivatives on positions
    D_y_N = torch.zeros((n + m - 1, m, d),
                        dtype=torch.float,
                        device=sample.device)
    D_y_N[:n, :, :] = B_0.reshape(n, m, 1) * dist_grad / zeta

    # Fill diagonal indices for position derivatives
    for i in range(m - 1):
        D_y_N[n + i, i, :] = torch.sum(D_y_N[:n, i, :], dim=0)

    # Calculate derivatives using Einstein summation
    D_nu_dv = torch.einsum("ij,jk->ik", SIM, D_nu_N)
    D_y_dv = torch.einsum("ij,jkl->ikl", SIM, D_y_N)

    # Calculate alpha-beta terms
    D_nu_alpha_beta = torch.einsum("kj,kji->kji", B,
                                   (D_nu_dv[:n, :].reshape(n, 1, m) +
                                    D_nu_dv[n:, :].reshape(1, m - 1, m)))

    D_y_alpha_beta = torch.einsum("ij,ijkl->ijkl", B,
                                  (D_y_dv[:n, :].reshape(n, 1, m, d) +
                                   D_y_dv[n:, :].reshape(1, m - 1, m, d)))

    # Calculate gradient with respect to weights
    D_nu_obj = torch.sum(dist * (B_0 / target_weight.reshape(1, -1)), dim=0)
    D_nu_obj += torch.einsum("ij,ijk->k", dist[:, :-1], D_nu_alpha_beta) / zeta
    D_nu_obj += torch.sum(
        (dist[:, -1] * B_0[:, -1]).reshape(-1, 1) * D_nu_dv[:n, :],
        dim=0) / zeta

    # Calculate gradient with respect to positions
    D_y_obj = torch.sum(dist_grad * ((1 - dist / zeta) * B_0).reshape(n, m, 1),
                        dim=0)
    D_y_obj += torch.einsum("ij,ijkl->kl", dist[:, :-1], D_y_alpha_beta) / zeta
    D_y_obj += torch.sum(
        (dist[:, -1] * B_0[:, -1]).reshape(-1, 1, 1) * D_y_dv[:n, :, :],
        dim=0) / zeta

    # Calculate entropy terms
    ln_B_1 = torch.zeros_like(B_0, dtype=torch.float)
    mask = B_0 != 0
    ln_B_1[mask] = torch.log(B_0[mask])
    ln_B_1 += -torch.log(sample_weight).reshape(
        -1, 1) - torch.log(target_weight) + 1

    # Calculate entropy gradients
    D_nu_entropy = torch.sum(
        (ln_B_1 - 1) * (B_0 / target_weight.reshape(1, -1)), dim=0) * zeta
    D_nu_entropy += torch.einsum("ij,ijk->k", ln_B_1[:, :-1], D_nu_alpha_beta)
    D_nu_entropy += torch.einsum("i,ij,i->j", ln_B_1[:, -1], D_nu_dv[:n, :],
                                 B_0[:, -1])

    D_y_entropy = torch.einsum("ij,ijkl->kl", ln_B_1[:, :-1], D_y_alpha_beta)
    D_y_entropy += torch.einsum("i,ijk,i->jk", ln_B_1[:, -1], D_y_dv[:n, :],
                                B_0[:, -1])
    D_y_entropy -= torch.einsum("ijk,ij,ij->jk", dist_grad, ln_B_1, B_0)

    # Add entropy terms to gradients
    D_nu_obj += D_nu_entropy
    D_nu_obj[:-1] -= D_nu_obj[-1]
    D_nu_obj[-1] = -torch.sum(D_nu_obj[:-1])

    D_y_obj += D_y_entropy

    # Calculate total cost
    total_cost = torch.sum(B_0 * dist) + zeta * torch.sum(ln_B_1 * B_0) - zeta

    return D_nu_obj, D_y_obj, total_cost


__all__ = [
    'uniform_sampler', 'iid_target_sampler_1d_uniform', 'euclidean_distance',
    'euc_dist_grad', 'wasserstein_distance', 'cont_wasserstein', 'gradient_EOT'
]
