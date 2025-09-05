#!/usr/bin/env python
'''
EDOT: Stereographic geometry

'''

import numpy as np


from . import EDOT_samplers as samplers



def dist_sphere_stereo(sample: np.ndarray,
                       target: np.ndarray,
                       exp: float = 2.) -> np.ndarray:
    '''
    Arc distance between points on sphere given stereographical
        coordinates

    Parameters
    ----------
    sample, target: numpy arrays of shape (n, 2) / (m, 2), points
    exp: float, power

    Return
    ------
    matrix of distances to power exp, numpy array of shape (n, m)
    '''
    sample = sample.reshape(-1, 2)
    target = target.reshape(-1, 2)
    sample_3d = samplers.inv_stereo_proj(sample).reshape(-1, 1, 3)
    target_3d = samplers.inv_stereo_proj(target).reshape(1, -1, 3)
    tmp = np.minimum(np.maximum(np.sum(sample_3d * target_3d, axis=2), -1), 1)

    return np.arccos(tmp) ** exp

def dist_sphere_natural(sample: np.ndarray,
                        target: np.ndarray,
                        exp: float = 2.) -> np.ndarray:
    '''
    Arc distance between points on sphere given natural coordinates
        as embedded in real vector space of dim 3

    Parameters
    ----------
    sample, target: numpy arrays of shape (n, 3) / (m, 3), points
    exp: float, power

    Return
    ------
    matrix of distances to power exp, numpy array of shape (n, m)
    '''
    sample_3d = sample.reshape(-1, 1, 3)
    target_3d = target.reshape(1, -1, 3)
    tmp = np.minimum(np.maximum(np.sum(sample_3d * target_3d, axis=2), -1), 1)

    return np.arccos(tmp) ** exp


def grad_dist_sphere_stereo(sample: np.ndarray,
                            target: np.ndarray,
                            exp: float = 2.) -> np.ndarray:
    '''
    Gradient of Arc Distances on sphere given stereographical coordinates

    Parameters
    ----------
    sample, target: numpy arrays of shape (n, 2) / (m, 2), points
    exp: float, power

    Return
    ------
    gradient of (distances to power exp), numpy array of shape (n, m, d)
    '''
    dist = dist_sphere_stereo(sample, target, exp) ** (1/exp)
    extra_coeff = exp * dist ** (exp - 1)
    grad = np.zeros(dist.shape + (2,), dtype=float)
    critical_ratio = np.ones_like(sample).reshape(-1, 1, 2) * target.reshape(1, -1, 2)
    critical_ratio = 2. / (1 + np.sum(critical_ratio ** 2, axis=2))

    sample_3d = samplers.inv_stereo_proj(sample).reshape(-1, 1, 3)
    target_3d = samplers.inv_stereo_proj(target).reshape(1, -1, 3)
    div_3d = -np.sqrt(np.maximum(0, (1 - np.sum(sample_3d*target_3d, axis=2) ** 2)))
    jacobian = np.zeros([target.shape[0], 2, 3], dtype=float)
    jacobian[:, :, 2] = 4 * target
    jacobian[:, 0, 1] = -4 * target[:, 0] * target[:, 1]
    jacobian[:, 1, 0] = jacobian[:, 0, 1]
    jacobian[:, 0, 0] = 2 * (1 - target[:, 0] ** 2 + target[:, 1] ** 2)
    jacobian[:, 1, 1] = 4 - jacobian[:, 0, 0]
    jacobian /= ((1 + np.sum(target ** 2, axis=1)) ** 2).reshape(-1, 1, 1)
    num_general = np.einsum("ik,jlk->ijl", sample_3d.reshape(-1, 3), jacobian)
    general = div_3d != 0
    special = div_3d == 0
    grad[general, :] = num_general[general, :] / div_3d[general].reshape(-1,1)

    identical = np.logical_and((np.sum(np.abs(sample_3d - target_3d), axis=2) <= 0.1), special)
    grad[identical, 0] = critical_ratio[identical]
    grad[identical, 1] = critical_ratio[identical]
    opposite = np.logical_xor(identical, special)
    grad[opposite, 0] = -critical_ratio[opposite]
    grad[opposite, 1] = -critical_ratio[opposite]

    return grad * extra_coeff.reshape(extra_coeff.shape + (1,))  # , grad
