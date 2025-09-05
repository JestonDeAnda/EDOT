#!/usr/bin/env python
"""
Sinkhorn Implementations in Numpy

"""
import numpy as np


def normalize(mat, axis_sum, axis=1):
    """
    A general normalizer of matrices.

    Parameters
    ----------
    mat     : numpy 2-array of size (n,m) (passed by reference, mutable)
    axis_sum: marginal sum, numpy 1-array, size matches mat and axis (mutable)
    axis    : flag of normalization direction, 1 for row-, 0 for column-

    P.S. normalize() will change both mat and axis_sum, please make copies first,
         such as, call normalize(mat.clone(), axis_sum.clone(), axis)

    Return
    ------
    normalized matrix in place (mat).
    """
    div = axis_sum / np.sum(mat, axis=axis)
    mat *= (div.reshape([-1, 1]) if axis else div)
    # tested, this kind of expression is better than `div.view([(1,-1),(-1,1)][axis])`
    # also `if axis` is faster than `if axis==0`
    return mat


def row_normalize(mat, row_sum):
    '''
    Row-normalization
    See `normalize()`
    '''
    div = row_sum / np.sum(mat, axis=1)
    mat *= div.reshape([-1, 1])
    return mat


def col_normalize(mat, col_sum):
    '''
    Column-normalization
    See `normalize()`
    '''
    div = col_sum / np.sum(mat, axis=0)
    mat *= div
    return mat


def sinkhorn_numpy_base(mat,
                        row_sum,
                        col_sum,
                        epsilon=1e-7,
                        max_iter=10000,):
    '''
    Sinkhorn scaling base

    Parameters
    ----------
    mat     : muted numpy 2-tensor of shape(n,m)
    row_sum : immuted numpy 1-tensor of size n
    col_sum : immuted numpy 1-tensor of size m
    epsilon : tolerance of 1-norm on column-sums with rows normalized
    max_iter: maximal iteration steps (multiples of 10)

    Return
    ------
    Sinkhorn scaled matrix (in place, can skip capturing it)
    '''

    diff = np.ones_like(col_sum, dtype=np.float64)
    max_iter //= 10

    while np.sum(np.abs(diff)) >= epsilon and max_iter:
        for i in range(10):
            row_normalize(col_normalize(mat, col_sum), row_sum)
        diff = col_sum - np.sum(mat, axis=0)
        max_iter -= 1

    return mat


def sinkhorn_numpy(mat,
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
    mat     : muted numpy 2-tensor of shape(n,m)
    row_sum : immuted numpy 1-tensor of size n
    col_sum : immuted numpy 1-tensor of size m
    epsilon : tolerance of 1-norm on column-sums with rows normalized
    max_iter: maximal iteration steps (multiples of 10)
    row_confident: whether row sums are all nonzeros (to bypass the check for nan-problem)
    col_confident: whether col sums are all nonzeros (to bypass the check for nan-problem)

    Return
    ------
    Sinkhorn scaled matrix (in place, can skip capturing it)
    '''
    shape_n, shape_m = mat.shape
    row_sum = row_sum if row_sum is not None else np.ones(shape_n, dtype=np.float64)
    col_sum = col_sum if col_sum is not None else np.ones(shape_m, dtype=np.float64)

    # if row_check:
    #     mat[row_sum != 0, col_sum != 0] = sinkhorn_numpy_base(mat[row_sum != 0, col_sum != 0],
    #                                                           row_sum[row_sum != 0],
    #                                                           col_sum[col_sum != 0],
    #                                                           epsilon, max_iter)
    # mat[:, col_sum == 0.] = 0.
    # mat[row_sum == 0., :] = 0.
    mat[:, col_sum != 0.] = sinkhorn_numpy_base(mat[:, col_sum != 0.],
                                                row_sum,
                                                col_sum[col_sum != 0],
                                                epsilon, max_iter)
    mat[:, col_sum == 0.] = 0.

    return mat


def roc_scbi_single_col(mat, index=0):
    """
    mat : matrix to calculate, 2d numpy array
    index: integer indicating index
    """
    n_row, n_col = mat.shape
    tmp = mat.clone()
    col = tmp[:, index].reshape([-1, 1])
    tmp /= col
    tmp = col_normalize(tmp, np.ones(n_col, dtype=np.float64))
    return -np.mean(np.log(tmp), axis=0) - np.log(n_row)

if __name__ == '__main__':
    pass
