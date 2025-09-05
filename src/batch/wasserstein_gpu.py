"""
Wasserstein on GPU
"""
# import ot  # pip install POT
import torch
from .sinkhorn_torch import sinkhorn_torch, sinkhorn_batch


def euclidean_distance(start: torch.Tensor,
                       end: torch.Tensor,
                       exp: float = 2.) -> torch.Tensor:
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
    # return cdist(start, end, "minkowski", p=exp)**exp
    if len(start.shape) == 1:
        start = start.reshape(-1, 1)
        end = end.reshape(-1, 1)
    n, m, d = start.shape[-2], end.shape[-2], end.shape[-1]
    shape_start = list(start.shape[:-2])
    shape_end = list(end.shape[:-2])
    ret = torch.absolute(start.reshape(shape_start + [1] * len(shape_end) + [n, 1, d]) - 
                         end.reshape([1] * len(shape_start) + shape_end + [1, m, d]))**exp
    ret = torch.sum(ret, dim=-1)
    return ret
    # we calculate `(distance ** exp)`
    # return ret ** (1 / exp)


def euc_dist_grad(start: torch.Tensor,
                  end: torch.Tensor,
                  exp: float = 2.) -> torch.Tensor:
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
    ret = torch.absolute(tmp)**(exp - 1)  # all >= 0
    ret[tmp > 0] *= -1
    return exp * ret


def wasserstein_distance(
        sample: torch.Tensor,
        sample_weight: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor,
        zeta: float = 0.002,
        distance: callable = euclidean_distance,
        distance_gradient: callable = euc_dist_grad) -> torch.Tensor:
    '''
    Sinkhorn Information Matrix

    Parameters
    ----------
    sample: torch tensor of samples, of shape (sample_size, space_dimension)
    sample_weight: torch tensor of sample weights, of shape (sample_size)
    target: torch tensor of samples, of shape (target_size, space_dimension)
    target_weight: torch tensor of target weights, of shape (target_size)
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

    dist = distance(sample, target, 2)  # get distance
    dist_grad = distance_gradient(sample, target, 2)

    ref = - dist / zeta
    ref -= (ref.max() + ref.min())/2
    mat = torch.exp(-ref)
    B_0 = sinkhorn_torch(mat, target_weight, sample_weight)


    # B_0 = ot.sinkhorn(sample_weight, target_weight, dist, zeta)

    return torch.sum(B_0 * dist)

def wasserstein_distance_batch(
        sample: torch.Tensor,
        sample_weight: torch.Tensor,
        target: torch.Tensor,
        target_weight: torch.Tensor,
        distance: callable = euclidean_distance,
        distance_gradient: callable = euc_dist_grad) -> torch.Tensor:
    '''
    Sinkhorn Information Matrix

    Parameters
    ----------
    sample: torch tensor of samples, of shape (sample_size, space_dimension)
    sample_weight: torch tensor of sample weights, of shape (sample_size)
    target: torch tensor of samples, of shape (target_size, space_dimension)
    target_weight: torch tensor of target weights, of shape (target_size)
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

    dist = distance(sample, target, 2)  # get distance
    # dist_grad = distance_gradient(sample, target, 2)

    ref = - dist
    partial_shape = list(ref.shape[:-2])
    reshaped_ref = ref.reshape(partial_shape+[-1])

    _max = reshaped_ref.max(dim=-1)[0]
    _min = reshaped_ref.min(dim=-1)[0]
    _scale = 100 / (_max - _min)

    ref -= ((_max+_min)/2).view(partial_shape+[1,1])
    ref *= _scale.view(partial_shape+[1,1])

    mat = torch.exp(-ref)

    B_0 = sinkhorn_batch(mat, target_weight, sample_weight)


    # B_0 = ot.sinkhorn(sample_weight, target_weight, dist, zeta)

    print(B_0.shape)
    return torch.sum(B_0.view(partial_shape+[-1]) * dist.view(partial_shape+[-1]), axis=-1)
