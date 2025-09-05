#!/usr/bin/env python
'''
EDOT_discretization

Some basic utility codes used in ODOT. Requires
`numpy` and `sinkhorn_numpy`.

Date: 2021-1-23
'''

import numpy as np
from scipy.spatial.distance import cdist

from . import sinkhorn_numpy as sk

uniform_sampler = lambda size: (
    np.linspace(0, 1 - (1. / size), num=size, endpoint=True).reshape(-1, 1) +
    (0.5 / size), np.ones(size, dtype=float) / size)


def construct_transport_matrix(n, m, k, zeta, sample, target, exp, distance,
                               distance_gradient):
    dist = distance(sample, target, exp)  # get distance
    dist_grad = distance_gradient(sample, target, exp)
    if exp != k:
        dist_grad *= ((k / exp) * (dist**(k / exp - 1))).reshape(n, m, 1)
        LM = -(dist)**(k / exp) / zeta
    else:
        LM = -dist / zeta
    LM -= np.max(LM)
    M = np.exp(LM)
    return dist, dist_grad, M


def iid_target_sampler_1d_uniform(target_size: int):
    '''
    Sample the iid samples from 1d uniform distribution

    Parameters
    ----------
    target_size: int

    Return
    ------
    (position, weights):
        position: numpy array of shape (target_size, 1) with values in [0,1].
        weights: numpy array of shape (target_size) of sum 1.
    '''
    target_pos = np.random.rand(target_size, 1)
    # target_weight = np.ones(target_size, dtype=float) / target_size
    target_weight = np.random.dirichlet([
        10.,
    ] * target_size)
    return target_pos, target_weight


# ===========================================================================


def euclidean_distance(start: np.ndarray,
                       end: np.ndarray,
                       exp: float = 2.) -> np.ndarray:
    '''
    Euclidean distance between two points

    Parameters
    ----------
    start: array of n starting points, `start.shape == (n, d)`
    end: array of m ending points, `end.shape == (m, d)`
    exp: exponent, float

    Return
    ------
    distance matrix of size n*m
    '''
    return cdist(start, end, "minkowski", p=exp)**exp
    if len(start.shape) == 1:
        start = start.reshape(-1, 1)
        end = end.reshape(-1, 1)
    n, m, d = start.shape[0], end.shape[0], end.shape[1]
    ret = np.absolute(start.reshape(n, 1, d) - end.reshape(1, m, d))**exp
    ret = np.sum(ret, axis=2)
    return ret
    # we calculate `(distance ** exp)`
    # return ret ** (1 / exp)


def euc_dist_grad(start: np.ndarray,
                  end: np.ndarray,
                  exp: float = 2.) -> np.ndarray:
    '''
    `exp`-distance between two arrays of points
    differentiated on `end`
    Parameters
    ----------
    start: array of n starting points, `start.shape == (n, d)`
    end: array of m ending points, `end.shape == (m, d)`
    exp: exponent, float

    Return
    ------
    distance matrix of size n*m
    '''
    if len(start.shape) == 1:
        start = start.reshape(-1, 1)
        end = end.reshape(-1, 1)
    n, m, d = start.shape[0], end.shape[0], end.shape[1]
    tmp = start.reshape(n, 1, d) - end.reshape(1, m, d)
    ret = np.absolute(tmp)**(exp - 1)  # all >= 0
    ret[tmp > 0] *= -1
    return exp * ret


# ===========================================================================


def wasserstein_distance(
        sample: np.ndarray,
        sample_weight: np.ndarray,
        target: np.ndarray,
        target_weight: np.ndarray,
        k: float = 2.,
        exp: float = 2.,
        zeta: float = 0.1,
        distance: callable = euclidean_distance,
        distance_gradient: callable = euc_dist_grad) -> np.ndarray:
    '''
    Sinkhorn Information Matrix

    Parameters
    ----------
    sample: numpy nd array of samples, of shape (sample_size, space_dimension)
    sample_weight: numpy ndarray of sample weights, of shape (sample_size)
    target: numpy nd array of samples, of shape (target_size, space_dimension)
    target_weight: numpy ndarray of target weights, of shape (target_size)
    k: float, the parameter in k-Wasserstein distance W_k^k
    exp: float, exponent of the exp-norm calculation as distance matrix
    zeta: float, the parameter in EOT
    distance: function that calculate the matrix of distance
    distance_gradient: function that calculate the tensor of gradient of distance

    Return
    ------
    matrix of shape (sample_size + target_size, sample_size + target_size),
        the Sinkhorn Information Matrix
    '''
    if len(sample.shape) == 1:
        sample = sample.reshape(-1, 1)
        target = target.reshape(-1, 1)
    n, m, d = sample.shape[0], target.shape[0], sample.shape[1]

    dist, dist_grad, M = construct_transport_matrix(n, m, k, zeta, sample,
                                                    target, exp, distance,
                                                    distance_gradient)
    cost = dist if exp == k else dist**(k / exp)

    B_0 = sk.sinkhorn_numpy(M, col_sum=target_weight, row_sum=sample_weight)

    return np.sum(B_0 * cost)


def cont_wasserstein(sampler: callable,
                     target: np.ndarray,
                     target_weight: np.ndarray,
                     N: int = 400,
                     ratio: int = 2,
                     distance: callable = wasserstein_distance,
                     zeta: float = 0.01,
                     k: float = 2.,
                     exp: float = 2.) -> tuple:
    '''
    Calculate Wasserstein distance of continuous
        using Richardson Extrapolation

    Parameters
    ----------
    sampler: (size: int) -> (positions, weights)
    target: numpy array of shape (m, d)
    target_weight: numpy array of shape (m)
    N: sample size in Richardson extrapolation
    ratio: ratio of sample sizes used in RE
    (other args): same as before.

    Return
    ------
    (Wasserstein-distance, [W(N), W(2 * N)])
    '''
    sample, sample_weight = sampler(N)

    W_1 = distance(sample,
                   sample_weight,
                   target,
                   target_weight,
                   zeta=zeta,
                   k=k,
                   exp=exp)
    sample, sample_weight = sampler(N * ratio)
    W_2 = distance(sample,
                   sample_weight,
                   target,
                   target_weight,
                   zeta=zeta,
                   k=k,
                   exp=exp)
    t = k / target.shape[1]
    W_inf = (W_2 * ratio**t - W_1) / (ratio**t - 1)
    return W_inf, [W_1, W_2]


# ===========================================================================


def gradient_wasserstein(
        sample: np.ndarray,
        sample_weight: np.ndarray,
        target: np.ndarray,
        target_weight: np.ndarray,
        k: float = 2.,
        exp: float = 2.,
        zeta: float = 0.1,
        distance: callable = euclidean_distance,
        distance_gradient: callable = euc_dist_grad) -> np.ndarray:
    '''
    Gradient of the Wasserstein

    Parameters
    ----------
    sample: numpy nd array of samples, of shape (sample_size, space_dimension)
    sample_weight: numpy ndarray of sample weights, of shape (sample_size)
    target: numpy nd array of samples, of shape (target_size, space_dimension)
    target_weight: numpy ndarray of target weights, of shape (target_size)
    k: float, the parameter in k-Wasserstein distance W_k^k
    exp: float, exponent of the exp-norm calculation as distance matrix
    zeta: float, the parameter in EOT
    distance: function that calculate the matrix of distance
    distance_gradient: function that calculate the tensor of gradient of distance

    Return
    ------
    matrix of shape (sample_size + target_size, sample_size + target_size),
        the Sinkhorn Information Matrix
    '''
    if len(sample.shape) == 1:
        sample = sample.reshape(-1, 1)
        target = target.reshape(-1, 1)
    n, m, d = sample.shape[0], target.shape[0], sample.shape[1]

    dist, dist_grad, M = construct_transport_matrix(n, m, k, zeta, sample,
                                                    target, exp, distance,
                                                    distance_gradient)

    # print(M, )
    B_0 = sk.sinkhorn_numpy(M, col_sum=target_weight, row_sum=sample_weight)
    B = B_0[:, :-1]
    A = sample_weight.copy()  # should be treated as a diagonal matrix
    D = target_weight[:-1].copy()

    # Construction of SIM
    SIM = np.zeros([n + m - 1, n + m - 1], dtype=float)
    C = B.T / A
    E = np.diag(D) - np.matmul(C, B)
    E = np.linalg.inv(E)
    SIM[n:, n:] = E
    SIM[n:, :n] = -np.matmul(E, C)
    SIM[:n, n:] = SIM[n:, :n].T
    SIM[:n, :n] = np.diag(1. / A) + np.matmul(np.matmul(C.T, E), C)

    # print(np.linalg.inv(SIM).round(5))  # SIM seems to be right!
    # =================================================================
    #  Question: would it be faster if I just call `np.linalg.inv`?
    # =================================================================
    SIM = SIM * zeta  # negative cancelled with IFT formula later
    # print((-SIM).__repr__())

    # Can we calculate the upper-triangular parts only?
    # The `SIM * zeta` can cause extra load which could have been canceled with
    # possible `Something / data` later. Be careful about this.

    # derivatives on $\nu$ and $y$
    # on $\nu$, the basis elements are taken as [0,...,0,1,0,...,0,-1]
    # D_nu_N.shape == (n+m-1, m-1)
    # The coordinates are first m-1 components of nu, with knowing
    #     the components of nu sum to 1. HOPE THIS IS CORRECT
    D_nu_N = np.zeros([n + m - 1, m - 1])
    D_nu_N[:n, :] = -B / D + B_0[:, -1].reshape(-1, 1) / target_weight[-1]
    #
    di = np.diag_indices(m - 1)
    # D_nu_N = np.concatenate((-B/D + B_0[:,-1].reshape(-1, 1) / target_weight[-1],
    #                          np.diag(1 - D)),
    #                         axis=0)

    # D_y_N.shape == (n+m-1, m, d) with d = sample.shape[1]
    D_y_N = np.zeros([n + m - 1, m, d], dtype=float)
    D_y_N[:n, :, :] = B_0.reshape(n, m, 1) * dist_grad / zeta
    # (again, careful about zeta)

    # alternative 1:1
    # tmp = np.sum(B.reshape(n, -1, 1) * dist_grad[:, :-1, :], axis=0)
    # D_y_N[n:, :, :][di] = tmp
    # altervative 1:2
    D_y_N[n:, :, :][di] = np.sum(D_y_N[:n, :m - 1, :], axis=0)
    # it is about 10x faster than using [np.fill_diagonal(...) for i in range(m-1)]

    # print(D_y_N.reshape(D_y_N.shape[:-1]).__repr__())

    D_nu_dv = np.einsum("ij,jk->ik", SIM, D_nu_N)
    D_y_dv = np.einsum("ij,jkl->ikl", SIM, D_y_N)
    # Should be correct, do some further check.

    # print(np.einsum("ij,jk->ik", SIM, alt).round(5))

    # D_nu_obj = np.sum(dist[:,:-1] * D_nu_N[:n,:], axis=0)  # Is this correct?
    D_nu_obj = np.sum(dist * (B_0 / target_weight), axis=0)
    D_nu_obj = D_nu_obj[:-1] - D_nu_obj[-1]
    D_nu_obj += np.einsum("kj,kji,kj->i", dist[:, :-1],
                          (D_nu_dv[:n, :].reshape(n, 1, m - 1) +
                           D_nu_dv[n:, :].reshape(1, m - 1, m - 1)), B) / zeta
    D_nu_obj += np.sum(
        (dist[:, -1] * B_0[:, -1]).reshape(-1, 1) * D_nu_dv[:n, :],
        axis=0) / zeta
    # `D_nu_obj` matches the empirical ones

    D_y_obj = np.sum(dist_grad * ((1 - dist / zeta) * B_0).reshape(n, m, 1),
                     axis=0)
    D_y_obj += np.einsum("kj,kjil,kj->il", dist[:, :-1],
                         (D_y_dv[:n, :, :].reshape(n, 1, m, d) +
                          D_y_dv[n:, :, :].reshape(1, m - 1, m, d)), B) / zeta
    D_y_obj += np.sum(
        (dist[:, -1] * B_0[:, -1]).reshape(-1, 1, 1) * D_y_dv[:n, :, :],
        axis=0) / zeta
    # print("D_ν Ω:",D_nu_obj)

    return (
        np.concatenate((D_nu_obj, [-np.sum(D_nu_obj)]), axis=0),
        D_y_obj,
        np.sum(B_0 * dist),
    )


def gradient_EOT(sample: np.ndarray,
                 sample_weight: np.ndarray,
                 target: np.ndarray,
                 target_weight: np.ndarray,
                 k: float = 2.,
                 exp: float = 2.,
                 zeta: float = 0.1,
                 distance: callable = euclidean_distance,
                 distance_gradient: callable = euc_dist_grad) -> np.ndarray:
    '''
    Gradient of the EOT total cost

    Parameters
    ----------
    sample: numpy nd array of samples, of shape (sample_size, space_dimension)
    sample_weight: numpy ndarray of sample weights, of shape (sample_size)
    target: numpy nd array of samples, of shape (target_size, space_dimension)
    target_weight: numpy ndarray of target weights, of shape (target_size)
    k: float, the parameter in k-Wasserstein distance W_k^k
    exp: float, exponent of the exp-norm calculation as distance matrix
    zeta: float, the parameter in EOT
    distance: function that calculate the matrix of distance
    distance_gradient: function that calculate the tensor of gradient of distance

    Return
    ------
    matrix of shape (sample_size + target_size, sample_size + target_size),
        the Sinkhorn Information Matrix
    '''
    if len(sample.shape) == 1:
        sample = sample.reshape(-1, 1)
        target = target.reshape(-1, 1)
    n, m, d = sample.shape[0], target.shape[0], sample.shape[1]

    dist, dist_grad, M = construct_transport_matrix(n, m, k, zeta, sample,
                                                    target, exp, distance,
                                                    distance_gradient)

    B_0 = sk.sinkhorn_numpy(M, col_sum=target_weight, row_sum=sample_weight)
    B = B_0[:, :-1]
    A = sample_weight.copy()  # should be treated as a diagonal matrix
    D = target_weight[:-1].copy()

    # Construction of SIM
    SIM = np.zeros([n + m - 1, n + m - 1], dtype=float)
    C = B.T / A
    E = np.diag(D) - np.matmul(C, B)
    E = np.linalg.inv(E)
    SIM[n:, n:] = E
    SIM[n:, :n] = -np.matmul(E, C)
    SIM[:n, n:] = SIM[n:, :n].T
    SIM[:n, :n] = np.diag(1. / A) + np.matmul(np.matmul(C.T, E), C)

    # print(np.linalg.inv(SIM).round(5))  # SIM seems to be right!
    # =================================================================
    #  Question: would it be faster if I just call `np.linalg.inv`?
    # =================================================================
    SIM = SIM * zeta  # negative cancelled with IFT formula later
    # print((-SIM).__repr__())

    # Can we calculate the upper-triangular parts only?
    # The `SIM * zeta` can cause extra load which could have been canceled with
    # possible `Something / data` later. Be careful about this.

    # derivatives on $\nu$ and $y$
    # on $\nu$, the basis elements are taken as [0,...,0,1,0,...,0,-1]
    # D_nu_N.shape == (n+m-1, m-1)
    # The coordinates are first m-1 components of nu, with knowing
    #     the components of nu sum to 1. HOPE THIS IS CORRECT
    D_nu_N = np.zeros([n + m - 1, m])
    D_nu_N[:n, :] = -B_0 / target_weight
    #
    di = np.diag_indices(m - 1)
    # D_nu_N = np.concatenate((-B/D + B_0[:,-1].reshape(-1, 1) / target_weight[-1],
    #                          np.diag(1 - D)),
    #                         axis=0)

    # D_y_N.shape == (n+m-1, m, d) with d = sample.shape[1]
    D_y_N = np.zeros([n + m - 1, m, d], dtype=float)
    D_y_N[:n, :, :] = B_0.reshape(n, m, 1) * dist_grad / zeta
    # (again, careful about zeta)

    # alternative 1:1
    # tmp = np.sum(B.reshape(n, -1, 1) * dist_grad[:, :-1, :], axis=0)
    # D_y_N[n:, :, :][di] = tmp
    # altervative 1:2
    D_y_N[n:, :, :][di] = np.sum(D_y_N[:n, :m - 1, :], axis=0)
    # it is about 10x faster than using [np.fill_diagonal(...) for i in range(m-1)]

    # print(D_y_N.reshape(D_y_N.shape[:-1]).__repr__())

    D_nu_dv = np.einsum("ij,jk->ik", SIM, D_nu_N)
    D_y_dv = np.einsum("ij,jkl->ikl", SIM, D_y_N)
    # Should be correct, do some further check.

    # print(np.einsum("ij,jk->ik", SIM, alt).round(5))

    D_nu_alpha_beta = np.einsum("kj,kji->kji", B,
                                (D_nu_dv[:n, :].reshape(n, 1, m) +
                                 D_nu_dv[n:, :].reshape(1, m - 1, m)))

    D_y_alpha_beta = np.einsum("ij,ijkl->ijkl", B,
                               (D_y_dv[:n, :].reshape(n, 1, m, d) +
                                D_y_dv[n:, :].reshape(1, m - 1, m, d)))

    # D_nu_obj = np.sum(dist[:,:-1] * D_nu_N[:n,:], axis=0)  # Is this correct?
    D_nu_obj = np.sum(dist * (B_0 / target_weight), axis=0)
    D_nu_obj += np.einsum("ij,ijk->k", dist[:, :-1], D_nu_alpha_beta) / zeta
    D_nu_obj += np.sum(
        (dist[:, -1] * B_0[:, -1]).reshape(-1, 1) * D_nu_dv[:n, :],
        axis=0) / zeta
    # `D_nu_obj` matches the empirical ones

    D_y_obj = np.sum(dist_grad * ((1 - dist / zeta) * B_0).reshape(n, m, 1),
                     axis=0)
    D_y_obj += np.einsum("ij,ijkl->kl", dist[:, :-1], D_y_alpha_beta) / zeta
    D_y_obj += np.sum(
        (dist[:, -1] * B_0[:, -1]).reshape(-1, 1, 1) * D_y_dv[:n, :, :],
        axis=0) / zeta
    # print("D_ν Ω:",D_nu_obj)

    ln_B_1 = np.zeros_like(B_0, dtype=float)
    ln_B_1[B_0 != 0] = np.log(B_0[B_0 != 0])
    ln_B_1 += -np.log(sample_weight).reshape(-1, 1) - np.log(target_weight) + 1
    D_nu_entropy = np.sum((ln_B_1 - 1) * (B_0 / target_weight), axis=0) * zeta
    D_nu_entropy += np.einsum("ij,ijk->k", ln_B_1[:, :-1], D_nu_alpha_beta)
    D_nu_entropy += np.einsum("i,ij,i->j", ln_B_1[:, -1], D_nu_dv[:n, :],
                              B_0[:, -1])

    D_y_entropy = np.einsum("ij,ijkl->kl", ln_B_1[:, :-1], D_y_alpha_beta)
    D_y_entropy += np.einsum("i,ijk,i->jk", ln_B_1[:, -1], D_y_dv[:n, :],
                             B_0[:, -1])
    D_y_entropy -= np.einsum("ijk,ij,ij->jk", dist_grad, ln_B_1, B_0)

    # print(D_nu_obj, D_y_obj)
    # should be the same as results of `gradient_wasserstein`

    D_nu_obj += D_nu_entropy
    D_nu_obj[:-1] -= D_nu_obj[-1]
    D_nu_obj[-1] = -np.sum(D_nu_obj[:-1])

    D_y_obj += D_y_entropy

    return (
        D_nu_obj,
        D_y_obj,
        np.sum(B_0 * dist) + zeta * np.sum(ln_B_1 * B_0) - zeta,
    )


# ==============================================================================
# def gradient_POS(sample: np.ndarray,
#                  sample_weight: np.ndarray,
#                  target: np.ndarray,
#                  target_weight: np.ndarray,
#                  k: float = 2.,
#                  exp: float = 2.,
#                  zeta: float = 0.1,
#                  distance: callable = euclidean_distance,
#                  distance_gradient: callable = euc_dist_grad) -> np.ndarray:
#     '''
#     Gradient of the EOT total cost

#     Parameters
#     ----------
#     sample: numpy nd array of samples, of shape (sample_size, space_dimension)
#     sample_weight: numpy ndarray of sample weights, of shape (sample_size)
#     target: numpy nd array of samples, of shape (target_size, space_dimension)
#     target_weight: numpy ndarray of target weights, of shape (target_size)
#     k: float, the parameter in k-Wasserstein distance W_k^k
#     exp: float, exponent of the exp-norm calculation as distance matrix
#     zeta: float, the parameter in EOT
#     distance: function that calculate the matrix of distance
#     distance_gradient: function that calculate the tensor of gradient of distance

#     Return
#     ------
#     matrix of shape (sample_size + target_size, sample_size + target_size),
#         the Sinkhorn Information Matrix
#     '''
#     if len(sample.shape) == 1:
#         sample = sample.reshape(-1, 1)
#         target = target.reshape(-1, 1)
#     n, m, d = sample.shape[0], target.shape[0], sample.shape[1]

#     dist = distance(sample, target, exp)  # get distance
#     dist_grad = distance_gradient(sample, target, exp)
#     if exp != k:
#         dist_grad *= ((k / exp) * (dist ** (k / exp - 1))).reshape(n, m, 1)
#         M = np.exp(-(dist) ** (k / exp) / zeta)
#     else:
#         M = np.exp(-dist / zeta)

#     B_0 = sk.sinkhorn_numpy(M, col_sum=target_weight, row_sum=sample_weight)
#     B = B_0[:, :-1]
#     A = sample_weight.copy()  # should be treated as a diagonal matrix
#     D = target_weight[:-1].copy()

#     # Construction of SIM
#     SIM = np.zeros([n + m - 1, n + m - 1], dtype = float)
#     C = B.T / A
#     E = np.diag(D) - np.matmul(C, B)
#     E = np.linalg.inv(E)
#     SIM[n:, n:] = E
#     SIM[n:, :n] = -np.matmul(E, C)
#     SIM[:n, n:] = SIM[n:, :n].T
#     SIM[:n, :n] = np.diag(1./A) + np.matmul(np.matmul(C.T, E), C)

#     SIM = SIM * zeta  # negative cancelled with IFT formula later

#     di = np.diag_indices(m-1)

#     # D_y_N.shape == (n+m-1, m, d) with d = sample.shape[1]
#     D_y_N = np.zeros([n+m-1, m, d], dtype=float)
#     D_y_N[:n, :, :] = B_0.reshape(n, m, 1) * dist_grad / zeta
#     # (again, careful about zeta)

#     D_y_N[n:, :, :][di] = np.sum(D_y_N[:n, :m-1, :],
#                                  axis=0)

#     D_y_dv = np.einsum("ij,jkl->ikl", SIM, D_y_N)

#     D_y_alpha_beta = np.einsum("ij,ijkl->ijkl", B,
#                                (D_y_dv[:n, :].reshape(n, 1, m, d) +
#                                 D_y_dv[n:, :].reshape(1, m-1, m, d)))

#     D_y_obj = np.sum(dist_grad * ((1 - dist / zeta) * B_0).reshape(n, m, 1),
#                      axis=0)
#     D_y_obj += np.einsum("ij,ijkl->kl", dist[:,:-1],
#                          D_y_alpha_beta) / zeta
#     D_y_obj += np.sum((dist[:, -1] * B_0[:, -1]).reshape(-1,1,1) * D_y_dv[:n, :, :],
#                       axis=0) / zeta
#     # print("D_ν Ω:",D_nu_obj)

#     ln_B_1 = np.ones_like(B_0, dtype=float)
#     ln_B_1[B_0 != 0] = np.log(B_0[B_0 != 0])
#     ln_B_1 += np.log(m*n)

#     D_y_entropy = np.einsum("ij,ijkl->kl", ln_B_1[:,:-1],
#                             D_y_alpha_beta)
#     D_y_entropy += np.einsum("i,ijk,i->jk", ln_B_1[:, -1], D_y_dv[:n, :], B_0[:, -1])
#     D_y_entropy -= np.einsum("ijk,ij,ij->jk", dist_grad, ln_B_1, B_0)

#     D_y_obj += D_y_entropy

#     likelihood = np.einsum("ij,ik->jk", B_0, B_0) - np.eye(m)
#     # print(likelihood, np.argmax(likelihood,axis=0))
#     diff = target-target[np.argmax(likelihood,axis=0)]
#     ddd = (euclidean_distance(diff, np.zeros([1,d])))
#     # print(diff, target, likelihood, B_0)
#     # raise
#     diff /= (ddd*15/(min(ddd)+1e-4))

#     # print(diff)

#     return (# D_nu_obj,
#             D_y_obj - diff,
#             np.sum(B_0 * dist) +
#             zeta * np.sum(ln_B_1 * B_0)
#             - zeta,
#            )


def gradient_POS(
    sample: np.ndarray,
    sample_weight: np.ndarray,
    target: np.ndarray,
    target_weight: np.ndarray,
    k: float = 2.,
    exp: float = 2.,
    zeta: float = 0.1,
    distance: callable = euclidean_distance,
    distance_gradient: callable = euc_dist_grad
) -> tuple[np.ndarray, float, np.ndarray]:
    '''
    Gradient of the EOT total cost (for target position and weight)

    Parameters
    ----------
    sample: ndarray (n, d)
    sample_weight: ndarray (n,)
    target: ndarray (m, d)
    target_weight: ndarray (m,)
    k, exp, zeta: scalar hyperparameters
    distance: function to compute distance matrix
    distance_gradient: function to compute distance gradient tensor

    Returns
    -------
    grad_target_position: ndarray (m, d)
    cost: scalar
    grad_target_weight: ndarray (m,)
    '''
    if len(sample.shape) == 1:
        sample = sample.reshape(-1, 1)
        target = target.reshape(-1, 1)
    n, m, d = sample.shape[0], target.shape[0], sample.shape[1]

    dist, dist_grad, M = construct_transport_matrix(n, m, k, zeta, sample,
                                                    target, exp, distance,
                                                    distance_gradient)

    B_0 = sk.sinkhorn_numpy(M, col_sum=target_weight, row_sum=sample_weight)

    B = B_0[:, :-1]
    A = sample_weight.copy()
    D = target_weight[:-1].copy()

    # SIM Construction
    SIM = np.zeros([n + m - 1, n + m - 1], dtype=float)
    C = B.T / A
    E = np.diag(D) - np.matmul(C, B)
    E = np.linalg.inv(E)
    SIM[n:, n:] = E
    SIM[n:, :n] = -np.matmul(E, C)
    SIM[:n, n:] = SIM[n:, :n].T
    SIM[:n, :n] = np.diag(1. / A) + np.matmul(C.T @ E, C)
    SIM *= zeta

    di = np.diag_indices(m - 1)
    D_y_N = np.zeros([n + m - 1, m, d], dtype=float)
    D_y_N[:n, :, :] = B_0.reshape(n, m, 1) * dist_grad / zeta
    D_y_N[n:, :, :][di] = np.sum(D_y_N[:n, :m - 1, :], axis=0)

    D_y_dv = np.einsum("ij,jkl->ikl", SIM, D_y_N)
    D_y_alpha_beta = np.einsum("ij,ijkl->ijkl", B,
                               (D_y_dv[:n, :].reshape(n, 1, m, d) +
                                D_y_dv[n:, :].reshape(1, m - 1, m, d)))

    D_y_obj = np.sum(dist_grad * ((1 - dist / zeta) * B_0).reshape(n, m, 1),
                     axis=0)
    D_y_obj += np.einsum("ij,ijkl->kl", dist[:, :m - 1], D_y_alpha_beta) / zeta
    D_y_obj += np.sum(
        (dist[:, -1] * B_0[:, -1]).reshape(-1, 1, 1) * D_y_dv[:n, :, :],
        axis=0) / zeta

    ln_B_1 = np.ones_like(B_0, dtype=float)
    ln_B_1[B_0 != 0] = np.log(B_0[B_0 != 0])
    ln_B_1 += np.log(m * n)

    D_y_entropy = np.einsum("ij,ijkl->kl", ln_B_1[:, :m - 1], D_y_alpha_beta)
    D_y_entropy += np.einsum("i,ijk,i->jk", ln_B_1[:, -1], D_y_dv[:n, :],
                             B_0[:, -1])
    D_y_entropy -= np.einsum("ijk,ij,ij->jk", dist_grad, ln_B_1, B_0)

    D_y_obj += D_y_entropy

    # Regularization to prevent duplicate centers
    likelihood = np.einsum("ij,ik->jk", B_0, B_0) - np.eye(m)
    diff = target - target[np.argmax(likelihood, axis=0)]
    ddd = euclidean_distance(diff, np.zeros([1, d]))
    diff /= (ddd * 15 / (min(ddd) + 1e-4))

    grad_target_position = D_y_obj - diff

    # ========== 新增权重梯度 ==========
    grad_target_weight = np.sum(B_0, axis=0) - target_weight

    # ========== 计算损失 ==========
    cost = np.sum(B_0 * dist) + zeta * np.sum(ln_B_1 * B_0) - zeta

    return grad_target_position, cost, grad_target_weight


# ===========================================================================


def gradient_descent_v2(sample: np.ndarray,
                        sample_weight: np.ndarray,
                        target: np.ndarray,
                        target_weight: np.ndarray,
                        zeta: float = 0.01,
                        k: float = 2.,
                        exp: float = 2.,
                        lower_bounds: float = 0.,
                        upper_bounds: float = 1.):
    '''
    gradient descent Version 2

    Parameters
    ----------
    sample: numpy ndarray of shape (n, d) representing sample positions
    sample_weight: numpy ndarray of sample weights
    target: numpy ndarray of shape (m, d)
    target_weight: numpy ndarray of shape (m, )
    zeta: EOT regularizing factor
    k: float, parameter for Wasserstein-k-distance
    exp: float, exp-metric on space X
    lower_bounds: float or np.ndarray, lower bounds for X
    upper_bounds: upper bounds for X

    Return
    ------
    (target, target_weight, history_pos, history_wts, history_values):
         optimal target / target_weight
         log of positions / weights / values

    '''
    if isinstance(lower_bounds, (int, float)):
        lower_bounds = np.ones_like(target, dtype=float) * lower_bounds
    if isinstance(upper_bounds, (int, float)):
        upper_bounds = np.ones_like(target, dtype=float) * upper_bounds

    history_values = [
        np.inf,
    ]
    history_pos = [
        None,
    ]
    history_wts = [
        None,
    ]

    epsilon = 1e-5
    value_diff = 1.
    count = 0

    while value_diff >= epsilon and count < 1000:
        count += 1
        history_pos += [
            target.copy(),
        ]
        history_wts += [
            target_weight.copy(),
        ]
        dn, dy, value = gradient_EOT(sample,
                                     sample_weight,
                                     target,
                                     target_weight,
                                     zeta=zeta,
                                     k=k,
                                     exp=exp)
        # value_diff = history_values[-1] - value
        value_diff = np.sum(np.abs(dn)) + np.sum(np.abs(dy))
        history_values += [
            value,
        ]
        dy /= (2 + 0.1 * count)**(0.5)
        dn /= (2 + 0.1 * count)**(0.5)
        # ratio1 = np.zeros_like(upper_bounds, dtype=float)
        # ratio2 = np.zeros_like(upper_bounds, dtype=float)
        # ratio1[dy != 0] = (upper_bounds - target)[dy != 0] / dy[dy != 0]
        # ratio2[dy != 0] = (target - lower_bounds)[dy != 0] / dy[dy != 0]

        target -= dy  # / count
        target_weight -= dn  # / count

    history_pos += [
        target,
    ]
    history_wts += [
        target_weight,
    ]

    return target, target_weight, history_pos, history_wts, history_values


def SGD_v1(sampler: callable,
           sample_size: int,
           target: np.ndarray,
           target_weight: np.ndarray,
           zeta: float = 0.01,
           k: float = 2.,
           exp: float = 2.,
           lower_bounds: float = 0.,
           upper_bounds: float = 1.,
           max_step: int = 3000) -> tuple:
    '''
    gradient descent (deprecated)

    '''
    if isinstance(lower_bounds, (int, float)):
        lower_bounds = np.ones_like(target, dtype=float) * lower_bounds
    if isinstance(upper_bounds, (int, float)):
        upper_bounds = np.ones_like(target, dtype=float) * upper_bounds

    history_values = [
        np.inf,
    ]
    history_pos = [
        None,
    ]
    history_wts = [
        None,
    ]

    epsilon = 1e-5
    value_diff = 1.
    count = 0

    while value_diff >= epsilon and count < max_step:
        count += 1

        sample, sample_weight = sampler(sample_size)
        # sample_weight = np.ones(sample_size, dtype=float) / sample_size

        history_pos += [
            target.copy(),
        ]
        history_wts += [
            target_weight.copy(),
        ]
        dn, dy, value = gradient_EOT(sample,
                                     sample_weight,
                                     target,
                                     target_weight,
                                     zeta=zeta,
                                     k=k,
                                     exp=exp)
        # value_diff = history_values[-1] - value
        value_diff = np.sum(np.abs(dn)) + np.sum(np.abs(dy))
        history_values += [
            value,
        ]
        # dy *= 100/(100+count)
        # dn *= 100/(100+count)
        ratio1 = np.zeros_like(upper_bounds, dtype=float)
        ratio2 = np.zeros_like(upper_bounds, dtype=float)
        ratio1[dy != 0] = (upper_bounds - target)[dy != 0] / dy[dy != 0]
        ratio2[dy != 0] = (target - lower_bounds)[dy != 0] / dy[dy != 0]

        rate = (1 + 0.1 * count)**(-0.6) / 2
        target -= dy * rate
        target_weight -= dn * rate

    history_pos += [
        target,
    ]
    history_wts += [
        target_weight,
    ]

    return target, target_weight, history_pos, history_wts, history_values


def SGD_v2(sampler: callable,
           sample_size: int,
           target: np.ndarray,
           target_weight: np.ndarray,
           zeta: float = 0.01,
           k: float = 2.,
           exp: float = 2.,
           lower_bounds: float = 0.,
           upper_bounds: float = 1.,
           max_step: int = 3000,
           **args) -> tuple:
    '''
    gradient descent (deprecated)

    '''
    if isinstance(lower_bounds, (int, float)):
        lower_bounds = np.ones_like(target, dtype=float) * lower_bounds
    if isinstance(upper_bounds, (int, float)):
        upper_bounds = np.ones_like(target, dtype=float) * upper_bounds

    distance_gen = args["distance"] if "distance" in args.keys(
    ) else euclidean_distance
    grad_dist_gen = args["gradient"] if "gradient" in args.keys(
    ) else euc_dist_grad

    history_values = [
        np.inf,
    ]
    history_pos = [
        None,
    ]
    history_wts = [
        None,
    ]

    epsilon = 1e-5
    value_diff = 1.
    count = 0

    while value_diff >= epsilon and count < max_step:
        count += 1

        sample, sample_weight = sampler(sample_size)
        # sample_weight = np.ones(sample_size, dtype=float) / sample_size

        history_pos += [
            target.copy(),
        ]
        history_wts += [
            target_weight.copy(),
        ]
        dn, dy, value = gradient_EOT(sample,
                                     sample_weight,
                                     target,
                                     target_weight,
                                     zeta=zeta,
                                     k=k,
                                     exp=exp,
                                     distance=distance_gen,
                                     distance_gradient=grad_dist_gen)
        # value_diff = history_values[-1] - value
        value_diff = np.sum(np.abs(dn)) + np.sum(np.abs(dy))
        history_values += [
            value,
        ]
        rate = (1 + 0.5 * count)**(-0.6) / 2

        dy *= rate
        dn *= rate
        ratio1 = np.zeros_like(upper_bounds, dtype=float) + 2
        ratio2 = np.zeros_like(upper_bounds, dtype=float) + 2
        ratio1 = dy / (target - upper_bounds)
        ratio2 = dy / (target - lower_bounds)
        min_ratio = max(np.max(ratio1), np.max(ratio2))
        if min_ratio >= 1.:
            dy /= min_ratio * 2
            print(dy, target)

        ratio1 = np.zeros_like(dn, dtype=float) + 2
        ratio2 = np.zeros_like(dn, dtype=float) + 2
        ratio1 = dn / (target_weight - 1)
        ratio2 = dn / (target_weight - 0)
        min_ratio = max(np.max(ratio1), np.max(ratio2))
        if min_ratio >= 1.:
            dn /= min_ratio * 2
            print(dn, target_weight)

        target -= dy
        target_weight -= dn

    history_pos += [
        target,
    ]
    history_wts += [
        target_weight,
    ]

    return target, target_weight, history_pos, history_wts, history_values


def SGD_momentum(sampler: callable,
                 sample_size: int,
                 target: np.ndarray,
                 target_weight: np.ndarray,
                 zeta: float = 0.01,
                 k: float = 2.,
                 exp: float = 2.,
                 lower_bounds: float = 0.,
                 upper_bounds: float = 1.,
                 max_step: int = 3000,
                 eta: float = 0.2,
                 **args) -> tuple:
    '''
    gradient descent using Momentum

    Parameters
    ----------
    sampler: callable, sampler of continuous distribution
    sample_size: int, size of the mini batch
    target: position of target distribution
    target_weight: weights fo target distribution
    zeta: regularizer
    k: power in Wasserstein distance
    exp: style of distance, exp-distance
    lower_bounds: numpy array or float, lower bounds of cell
    upper_bounds: numpy array or float, upper bounds of cell
    max_step: int, the upper limit of the iterations
    eta: float, parameter of SGD momentum

    Return
    ------
    (result_position, result_weight, logs)

    '''
    if isinstance(lower_bounds, (int, float)):
        lower_bounds = np.ones_like(target, dtype=float) * lower_bounds
    if isinstance(upper_bounds, (int, float)):
        upper_bounds = np.ones_like(target, dtype=float) * upper_bounds

    # V2
    interval_len = np.sum(upper_bounds - lower_bounds)
    # V2
    dim = target.shape[-1]

    distance_gen = args["distance"] if "distance" in args.keys(
    ) else euclidean_distance
    grad_dist_gen = args["gradient"] if "gradient" in args.keys(
    ) else euc_dist_grad

    history_values = [
        np.inf,
    ]
    history_pos = [
        None,
    ]
    history_wts = [
        None,
    ]

    epsilon = 1e-4
    value_diff = 1.
    count = 0

    dn_o = np.zeros_like(target_weight)
    dy_o = np.zeros_like(target)

    while value_diff >= epsilon and count < max_step:
        count += 1

        sample, sample_weight = sampler(sample_size)
        # sample_weight = np.ones(sample_size, dtype=float) / sample_size

        history_pos += [
            target.copy(),
        ]
        history_wts += [
            target_weight.copy(),
        ]
        dn, dy, value = gradient_EOT(sample,
                                     sample_weight,
                                     target,
                                     target_weight,
                                     zeta=zeta,
                                     k=k,
                                     exp=exp,
                                     distance=distance_gen,
                                     distance_gradient=grad_dist_gen)
        dn_o = dn_o * eta + dn
        dy_o = dy_o * eta + dy * 3
        dn = dn_o.copy()
        dy = dy_o.copy()

        # V2
        # dy /= np.sum(np.abs(dy)) * (0.1 * dim * interval_len)
        # dn /= np.sum(np.abs(dn)) * (0.1)

        # value_diff = history_values[-1] - value
        value_diff = np.sum(np.abs(dn)) + np.sum(np.abs(dy))
        history_values += [
            value,
        ]
        rate = (1 + 0.2 * count)**(-0.5) / 2

        dy *= rate
        dn *= rate
        ratio1 = np.zeros_like(upper_bounds, dtype=float) + 2
        ratio2 = np.zeros_like(upper_bounds, dtype=float) + 2
        ratio1 = dy / (target - upper_bounds)
        ratio2 = dy / (target - lower_bounds)
        min_ratio = max(np.max(ratio1), np.max(ratio2))
        if min_ratio >= 1.:
            dy /= min_ratio * 2
            print(count, dy, target)

        ratio1 = np.zeros_like(dn, dtype=float) + 2
        ratio2 = np.zeros_like(dn, dtype=float) + 2
        ratio1 = dn / (target_weight - 1)
        ratio2 = dn / (target_weight - 0)
        min_ratio = max(np.max(ratio1), np.max(ratio2))
        if min_ratio >= 1.:
            dn /= min_ratio * 2
            print(count, dn, target_weight)

        target -= dy  # * 5
        target_weight -= dn

    history_pos += [
        target,
    ]
    history_wts += [
        target_weight,
    ]

    return target, target_weight, history_pos, history_wts, history_values


# ===========================================================================


def single_mc_stat_wasserstein(
        size: int,
        target_sampler: callable = iid_target_sampler_1d_uniform,
        sample_sampler: callable = uniform_sampler,
        volume: int = 10000,
        sample_size: int = 1000,
        zeta: float = 0.01,
        k: float = 2.,
        exp: float = 2.,
        ratio: int = 2) -> tuple:
    """
    Monte Carlo statistical tests on Wasserstein Distance

    Parameters
    ----------
    target_sampler(size:int)->(positions, weights):
            position: numpy array of shape (target_size, dim).
            weights: numpy array of shape (target_size) of sum 1.
    sample_sampler(size:int)->(positions, weights):
            position: numpy array of shape (sample_size, dim).
            weights: numpy array of shape (sample_size) of sum 1.
    sizes: iterable of int, the set/tuple/list of sizes


    Return
    ------
    (mean, std, five_pts_quantile)
    """
    result = {}
    log = np.zeros(volume, dtype=float)
    for i in range(volume):
        target, target_weight = target_sampler(size)
        log[i], _ = cont_wasserstein(sample_sampler,
                                     target,
                                     target_weight,
                                     N=sample_size,
                                     zeta=zeta,
                                     k=k,
                                     exp=exp,
                                     ratio=ratio)
        if not np.isfinite(log[i]):
            print(target)
    return (
        np.mean(log[np.isfinite(log)]),
        np.std(log[np.isfinite(log)]),
        np.quantile(log[np.isfinite(log)], [0.05, 0.25, 0.5, 0.75, 0.95]),
        len(log[np.isfinite(log)]),
    )


# ===============================================================

if __name__ == "__main__":
    pass
