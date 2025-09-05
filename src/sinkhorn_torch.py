#!/usr/bin/env python
"""
Sinkhorn Implementations in PyTorch

This module provides implementations of the Sinkhorn algorithm for optimal transport
problems using PyTorch. It includes functions for matrix normalization and Sinkhorn scaling
with support for both single matrices and batched operations.
"""
import torch


def normalize(mat, axis_sum, axis=-1):
    """
    A general normalizer of matrices.

    Parameters
    ----------
    mat     : torch tensor of size (..., n, m) (passed by reference, mutable)
    axis_sum: marginal sum, torch tensor, size matches mat and axis (mutable)
    axis    : flag of normalization direction, -1 for row-, -2 for column-normalization

    Note
    ----
    normalize() will change mat in-place. If you need to preserve the original matrix,
    make a copy first, e.g., normalize(mat.clone(), axis_sum.clone(), axis)

    Return
    ------
    normalized matrix in place (mat).
    """
    # Calculate sum along specified axis
    sum_along_axis = torch.sum(mat, dim=axis, keepdim=True)
    # Avoid division by zero by adding a small epsilon
    sum_along_axis = torch.clamp(sum_along_axis, min=1e-10)
    # Calculate scaling factors
    scaling = axis_sum.unsqueeze(axis) / sum_along_axis
    # Apply scaling
    mat *= scaling
    return mat


def row_normalize(mat, row_sum):
    '''
    Row-normalization for matrices of any dimension.
    
    Parameters
    ----------
    mat     : torch tensor of shape (..., n, m) (passed by reference, mutable)
    row_sum : torch tensor of shape (..., n) with target row sums
    
    Return
    ------
    Row-normalized matrix (in-place modification of mat)
    '''
    # Calculate current row sums
    sum_along_rows = torch.sum(mat, dim=-1, keepdim=True)
    # Avoid division by zero
    sum_along_rows = torch.clamp(sum_along_rows, min=1e-10)
    # Calculate scaling factors and reshape for broadcasting
    scaling = row_sum.unsqueeze(-1) / sum_along_rows
    # Apply scaling
    mat *= scaling
    return mat


def col_normalize(mat, col_sum):
    '''
    Column-normalization for matrices of any dimension.
    
    Parameters
    ----------
    mat     : torch tensor of shape (..., n, m) (passed by reference, mutable)
    col_sum : torch tensor of shape (..., m) with target column sums
    
    Return
    ------
    Column-normalized matrix (in-place modification of mat)
    '''
    # Calculate current column sums
    sum_along_cols = torch.sum(mat, dim=-2, keepdim=True)
    # Avoid division by zero
    sum_along_cols = torch.clamp(sum_along_cols, min=1e-10)
    # Calculate scaling factors and reshape for broadcasting
    scaling = col_sum.unsqueeze(-2) / sum_along_cols
    # Apply scaling
    mat *= scaling
    return mat


def row_normalize_fast(mat, row_sum):
    '''
    Fast row-normalization for matrices of any dimension without zero-checking.
    For internal use only when zero rows have been filtered out.
    
    Parameters
    ----------
    mat     : torch tensor of shape (..., n, m) (passed by reference, mutable)
    row_sum : torch tensor of shape (..., n) with target row sums
    
    Return
    ------
    Row-normalized matrix (in-place modification of mat)
    '''
    # Calculate current row sums
    sum_along_rows = torch.sum(mat, dim=-1, keepdim=True)
    # Calculate scaling factors and reshape for broadcasting
    scaling = row_sum.unsqueeze(-1) / sum_along_rows
    # Apply scaling
    mat *= scaling
    return mat


def col_normalize_fast(mat, col_sum):
    '''
    Fast column-normalization for matrices of any dimension without zero-checking.
    For internal use only when zero columns have been filtered out.
    
    Parameters
    ----------
    mat     : torch tensor of shape (..., n, m) (passed by reference, mutable)
    col_sum : torch tensor of shape (..., m) with target column sums
    
    Return
    ------
    Column-normalized matrix (in-place modification of mat)
    '''
    # Calculate current column sums
    sum_along_cols = torch.sum(mat, dim=-2, keepdim=True)
    # Calculate scaling factors and reshape for broadcasting
    scaling = col_sum.unsqueeze(-2) / sum_along_cols
    # Apply scaling
    mat *= scaling
    return mat


def sinkhorn_torch_base(
    mat,
    row_sum,
    col_sum,
    epsilon=1e-7,
    max_iter=10000,
):
    '''
    Sinkhorn scaling base algorithm implementation.

    Parameters
    ----------
    mat     : torch tensor of shape (..., n, m) (modified in-place)
    row_sum : torch tensor of shape (..., n) with target row sums
    col_sum : torch tensor of shape (..., m) with target column sums
    epsilon : tolerance of 1-norm on column-sums with rows normalized
    max_iter: maximal iteration steps (multiples of 10)

    Return
    ------
    Sinkhorn scaled matrix (modified in-place)
    '''
    # Initialize convergence tracking
    diff = torch.ones_like(col_sum, dtype=torch.float64)
    # Adjust max_iter to account for inner loop
    remaining_iter = max_iter // 10

    # Main Sinkhorn iteration loop
    while torch.sum(torch.abs(diff)) >= epsilon and remaining_iter > 0:
        # Inner loop for efficiency (10 iterations at a time)
        for _ in range(10):
            # Alternating row and column normalizations using fast versions
            # Since we've filtered zero rows/columns in the calling functions,
            # we can safely use the fast versions without zero-checking
            col_normalize_fast(mat, col_sum)
            row_normalize_fast(mat, row_sum)

        # Check convergence on column sums
        diff = col_sum - torch.sum(mat, dim=-2)
        remaining_iter -= 1

    return mat


def sinkhorn_torch(
    mat,
    row_sum=None,
    col_sum=None,
    epsilon=1e-7,
    max_iter=10000,
):
    '''
    Sinkhorn scaling algorithm for 2D matrices with handling of zero marginals.

    Parameters
    ----------
    mat     : torch tensor of shape (n, m) (modified in-place)
    row_sum : torch tensor of shape (n) with target row sums, defaults to uniform
    col_sum : torch tensor of shape (m) with target column sums, defaults to uniform
    epsilon : tolerance for convergence checking
    max_iter: maximum number of iterations (multiples of 10)

    Return
    ------
    Sinkhorn scaled matrix (modified in-place)
    '''
    # Get matrix dimensions
    shape_n, shape_m = mat.shape

    # Default to uniform distributions if marginals not provided
    row_sum = row_sum if row_sum is not None else torch.ones(
        shape_n, dtype=torch.float64)
    col_sum = col_sum if col_sum is not None else torch.ones(
        shape_m, dtype=torch.float64)

    # Handle zero row sums
    valid_rows = row_sum != 0.
    # Handle zero column sums
    valid_cols = col_sum != 0.

    # Only process valid rows and columns
    if torch.all(valid_rows) and torch.all(valid_cols):
        # If all marginals are valid, process the entire matrix
        mat = sinkhorn_torch_base(mat, row_sum, col_sum, epsilon, max_iter)
    elif torch.any(valid_rows) and torch.any(valid_cols):
        # Process only the submatrix with non-zero marginals
        mat[valid_rows][:, valid_cols] = sinkhorn_torch_base(
            mat[valid_rows][:, valid_cols], row_sum[valid_rows],
            col_sum[valid_cols], epsilon, max_iter)
        # Set rows with zero sum to zero
        mat[~valid_rows, :] = 0.
        # Set columns with zero sum to zero
        mat[:, ~valid_cols] = 0.
    else:
        # If all marginals are zero, set the entire matrix to zero
        mat.zero_()

    return mat


def sinkhorn_batch(
    mat,
    row_sum=None,
    col_sum=None,
    epsilon=1e-7,
    max_iter=10000,
):
    '''
    Batched Sinkhorn scaling algorithm for tensors with shape (..., n, m).

    Parameters
    ----------
    mat     : torch tensor of shape (..., n, m) (modified in-place)
    row_sum : torch tensor of shape (..., n) with target row sums, defaults to uniform
    col_sum : torch tensor of shape (..., m) with target column sums, defaults to uniform
    epsilon : tolerance for convergence checking
    max_iter: maximum number of iterations (multiples of 10)

    Return
    ------
    Sinkhorn scaled matrix (modified in-place)
    '''
    # Get matrix dimensions
    shape_n, shape_m = mat.shape[-2:]
    shape_remainder = list(mat.shape[:-2])

    # Default to uniform distributions if marginals not provided
    row_sum = row_sum if row_sum is not None else torch.ones(
        shape_remainder + [shape_n], dtype=torch.float64)
    col_sum = col_sum if col_sum is not None else torch.ones(
        shape_remainder + [shape_m], dtype=torch.float64)

    # Handle zero marginals in batched mode
    valid_rows = row_sum != 0.
    valid_cols = col_sum != 0.

    # Create a mask for valid entries
    batch_dims = len(shape_remainder)
    row_mask = valid_rows.view(shape_remainder + [shape_n, 1])
    col_mask = valid_cols.view(shape_remainder + [1, shape_m])
    valid_mask = torch.logical_and(row_mask, col_mask)

    # Only process valid entries
    if torch.all(valid_mask):
        # If all entries are valid, process the entire tensor
        mat = sinkhorn_torch_base(mat, row_sum, col_sum, epsilon, max_iter)
    elif torch.any(valid_mask):
        # Create a copy of the matrix to work with
        valid_mat = mat.clone()
        # Zero out invalid entries
        valid_mat = torch.where(valid_mask, valid_mat,
                                torch.zeros_like(valid_mat))
        # Apply Sinkhorn to valid entries
        valid_mat = sinkhorn_torch_base(valid_mat, row_sum, col_sum, epsilon,
                                        max_iter)
        # Update original matrix where mask is True
        mat = torch.where(valid_mask, valid_mat, torch.zeros_like(mat))
    else:
        # If no valid entries, set the entire tensor to zero
        mat.zero_()

    return mat


__all__ = [
    'normalize', 'row_normalize', 'col_normalize', 'sinkhorn_torch_base',
    'sinkhorn_torch', 'sinkhorn_batch'
]
