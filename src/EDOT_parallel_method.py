#!/usr/bin/env python
'''
EDOT: parallel methods

'''
import numpy as np



def structural_sample_kdtree(target_size: int,
                             sampler: callable,
                             sample_size: int,
                             sample_args: dict = {},
                             limits: tuple = (0., 1.),
                             size_thres: int = 6,
                             k: float = 2.,
                             square: bool = True) -> list:
    r'''
    Sample with certain structure. (kD-Tree version)
    (Algorithm 2 in the paper)

    Parameters
    ----------
    target_size: int, the size of the support of target distribution
    sampler: callable, iid sampler for continuous distribution $\mu$
    sample_size: int, the $N_0$, size of total samples to take
                 (at least 100 times target size)
    sample_args: dict, extra arguments (besides `size`) for the sampler
    limits: tuple, tuple of numpy arrays (of size `d`=dimension of space) or int
    size_thres: int = 6, the max number (included) allowed in each cell
    k: float = 2, the power of Wasserstein.
    square: controls the shape, True for making region of same size on all directions
            False for keeping the similar shape as original region.

    Return
    ------
    list of tasks suitable for multiprocessing maps
    '''
    sample, _ = sampler(sample_size, **sample_args)
    if len(sample.shape) == 1:
        sample.reshape(-1, 1)
    assert len(sample.shape) == 2
    dimension = sample.shape[-1]
    for i in range(2):
        if isinstance(limits[i], (int, float)):
            limits[i] = np.ones(dimension, dtype=float) * limits[i]

    root = [sample_size, target_size, sample, limits, []]
    job_list = []

    def dfs_gen(root_node: dict, level: int = 0,
                job_list: list = job_list,
                square: bool = True):
        '''
        Generate the tree using dfs
        '''
        vertex = root_node[3]
        if square:
            level = np.argmax(vertex[1] - vertex[0])
        # print(vertex)
        mid = (vertex[0][level % dimension] + vertex[1][level % dimension]) / 2
        sample_old = root_node[2]
        # a faster way of dividing samples:
        #    use binary tree (each component's > or <= mid)
        #    we may rewrite later if memory cost is too large
        # in which case, the limits will not be bounded by
        #    the constraint that lower smaller than larger
        for i in range(2):

            lower = vertex[0].copy()
            upper = vertex[1].copy()
            if i == 0:
                upper[level % dimension] = mid
                index = sample_old[:, level % dimension] < mid
            else:
                lower[level % dimension] = mid
                index = sample_old[:, level % dimension] >= mid

            sample = sample_old[index, :].copy()
            split_size = sample.shape[0]

            size = int(np.round(root_node[1] * (split_size)**(dimension/(dimension+k)) /
                                ((split_size)**(dimension/(dimension+k)) +
                                 (root_node[0] - split_size)**(dimension/(dimension+k)))))

            root_node[-1] += [[split_size,
                               size,
                               sample.copy(),
                               [lower.copy(), upper.copy()],
                               []], ]
            # print(root_node[-1][-1][:2], root_node[-1][-1][3])
        root_node[2] = None  # let GC work
        for node in root_node[-1]:
            if node[1] > size_thres:
                # divide in next level
                dfs_gen(node, level + 1, square=square)
            elif node[1] != 0:
                # leaf nodes
                np.random.shuffle(node[2])
                job_list += [node[:-1], ]

    dfs_gen(root, square)

    return root, job_list

def sampler_from_data(data: np.ndarray) -> callable:
    """
    Construct a sampler from reusing the data

    This returns a closure which works as a sampler
    """

    cur = 0

    def inner_sampler(size: int) -> tuple:
        """
        Inner one

        Return
        ------
        (sample, weight): sample numpy ndarray of shape (size, d),
                          weight numpy ndarray of shape (size, )
        """
        nonlocal cur
        cur += size
        weight = np.ones(size, dtype=float) / size
        if cur >= data.shape[0]:
            cur -= data.shape[0]
            sample = np.concatenate((data[cur - size:, :],
                                     data[:cur, :]),
                                    axis=0)
            np.random.shuffle(data)
        else:
            sample = data[cur - size: cur, :]

        return sample, weight

    return inner_sampler

if __name__ == '__main__':
    pass
