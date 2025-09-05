#!/usr/bin/env python
"""
Sinkhorn Implementations in PyTorch

"""
import torch

def normalize(mat, axis_sum, axis=-1):
    """
    A general normalizer of matrices.

    Parameters
    ----------
    mat     : torch 2-tensor of size (n,m) (passed by reference, mutable)
    axis_sum: marginal sum, torch 1-tensor, size matches mat and axis (mutable)
    axis    : flag of normalization direction, 1 for row-, 0 for column-

    P.S. normalize() will change both mat and axis_sum, please make copies first,
         such as, call normalize(mat.clone(), axis_sum.clone(), axis)

    Return
    ------
    normalized matrix in place (mat).
    """
    div = axis_sum / torch.sum(mat, dim=axis)
    mat *= (div.view([-1, 1]) if axis else div)
    return mat

def row_normalize(mat, row_sum):
    '''
    Row-normalization
    See `normalize()`
    '''
    div = row_sum / torch.sum(mat, dim=-1)
    mat *= div.view(list(mat.shape[:-1])+[1])
    return mat

def col_normalize(mat, col_sum):
    '''
    Column-normalization
    See `normalize()`
    '''
    div = col_sum / torch.sum(mat, dim=-2)
    mat *= div.view((list(mat.shape[:-2])+[1,-1]))
    return mat


def sinkhorn_torch_base(mat,
                        row_sum,
                        col_sum,
                        epsilon=1e-7,
                        max_iter=10000,):
    '''
    Sinkhorn scaling base

    Parameters
    ----------
    mat     : muted torch 2-tensor of shape(n,m)
    row_sum : immuted torch 1-tensor of size n
    col_sum : immuted torch 1-tensor of size m
    epsilon : tolerance of 1-norm on column-sums with rows normalized
    max_iter: maximal iteration steps (multiples of 10)

    Return
    ------
    Sinkhorn scaled matrix (in place, can skip capturing it)
    '''

    diff = torch.ones_like(col_sum, dtype=torch.float64)
    max_iter //= 10

    while torch.sum(torch.abs(diff)) >= epsilon and max_iter:
        for i in range(10):
            row_normalize(col_normalize(mat, col_sum), row_sum)
        diff = col_sum - torch.sum(mat, dim=-2)
        max_iter -= 1

    return mat


def sinkhorn_torch(mat,
                   row_sum=None,
                   col_sum=None,
                   epsilon=1e-7,
                   max_iter=10000,
                   # row_check=False, # Not activate for efficiency concern
                  ):
    '''
    Sinkhorn scaling base

    Parameters
    ----------
    mat     : muted torch 2-tensor of shape(n,m)
    row_sum : immuted torch 1-tensor of size n
    col_sum : immuted torch 1-tensor of size m
    epsilon : tolerance of 1-norm on column-sums with rows normalized
    max_iter: maximal iteration steps (multiples of 10)
    row_confident: whether row sums are all nonzeros (to bypass the check for nan-problem)
    col_confident: whether col sums are all nonzeros (to bypass the check for nan-problem)

    Return
    ------
    Sinkhorn scaled matrix (in place, can skip capturing it)
    '''
    shape_n, shape_m = mat.shape
    row_sum = row_sum if row_sum is not None else torch.ones(shape_n, dtype=torch.float64)
    col_sum = col_sum if col_sum is not None else torch.ones(shape_m, dtype=torch.float64)

    mat[:, col_sum != 0.] = sinkhorn_torch_base(mat[:, col_sum != 0.],
                                                row_sum,
                                                col_sum[col_sum != 0],
                                                epsilon, max_iter)
    mat[:, col_sum == 0.] = 0.

    return mat



def sinkhorn_batch(mat,
                   row_sum=None,
                   col_sum=None,
                   epsilon=1e-7,
                   max_iter=10000,
                   # row_check=False, # Not activate for efficiency concern
                  ):
    '''
    Sinkhorn scaling base

    Parameters
    ----------
    mat     : muted torch 2-tensor of shape(n,m)
    row_sum : immuted torch 1-tensor of size n
    col_sum : immuted torch 1-tensor of size m
    epsilon : tolerance of 1-norm on column-sums with rows normalized
    max_iter: maximal iteration steps (multiples of 10)
    row_confident: whether row sums are all nonzeros (to bypass the check for nan-problem)
    col_confident: whether col sums are all nonzeros (to bypass the check for nan-problem)

    Return
    ------
    Sinkhorn scaled matrix (in place, can skip capturing it)
    '''
    shape_n, shape_m = mat.shape[-2:]
    shape_remainder = list(mat.shape[:-2])
    row_sum = row_sum if row_sum is not None else torch.ones(shape_remainder + [shape_n], dtype=torch.float64)
    col_sum = col_sum if col_sum is not None else torch.ones(shape_remainder + [shape_m], dtype=torch.float64)

    return sinkhorn_torch_base(mat,
                               row_sum,
                               col_sum,
                               epsilon, max_iter)
    
