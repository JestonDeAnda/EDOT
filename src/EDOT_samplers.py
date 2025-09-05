#!/usr/bin/env python
'''
EDOT: Samplers

'''
import numpy as np
from scipy import stats as st


PARAMS = ((0.2, 0.1, 0.3), (0.7, 0.2, 0.7))
LIMITS = (0, 1)

# Gauss 1d

def tg_iid_sampler_1d(size: int,
                      params: tuple = PARAMS,
                      limits: tuple = LIMITS) -> tuple:
    '''
    Parameters
    ----------
    size: int, sample size.
    params: parameters, [(mean_1, std_1, weight_1), ...]

    Return
    ------
    (position, weights)
    '''
    position = np.zeros([size, 1], dtype=float)
    weights = np.ones(size, dtype=float) / size
    spec = np.random.multinomial(size, [x[2] for x in params])
    # print(spec)
    pointer = 0
    for i, size_p in enumerate(spec):
        mu, sigma = params[i][0], params[i][1]
        tmp = st.truncnorm((limits[0] - mu) / sigma,
                           (limits[1] - mu) / sigma,
                           loc=mu, scale=sigma).rvs(size_p)
        position[pointer : pointer+size_p, :] = tmp.copy().reshape(-1, 1)
        pointer += size_p

    return position, weights


def tg_even_sampler_1d(size: int,
                       params: tuple = PARAMS,
                       limits: tuple = LIMITS) -> tuple:
    '''
    Truncated Gaussian Sampler
    Samples are evenly positioned
    '''
    lower, upper = limits[0], limits[1]
    resolution = (upper - lower) / (2 * size)
    position = np.linspace(lower + resolution, upper - resolution, size)

    weight = np.zeros_like(position, dtype=float)
    for param in params:
        mean, sigma, prob = param

        wt_0 = st.truncnorm((lower - mean) / sigma,
                            (upper - mean) / sigma,
                            loc=mean, scale=sigma).pdf(position)
        wt_0 /= np.sum(wt_0)
        weight += prob * wt_0
        # print(weight.__repr__())
    return position.reshape(-1, 1), weight



# 2d uniform

def uniform_2d_iid(size: int,
                   lower: np.ndarray = np.array([0., 0.]),
                   upper: np.ndarray = np.array([0., 0.])) -> tuple:
    """
    Uniform 2d Sampler, IID
    """

    return (np.random.rand(size, 2) * (upper - lower) + lower,
            np.ones(size, dtype=float) / size)


# kd uniform

def uniform_kd_iid(size: int,
                   lower: np.ndarray = np.array([0., 0.]),
                   upper: np.ndarray = np.array([1., 1.])) -> tuple:
    """
    Uniform kd Sampler, IID
    """

    return (np.random.rand(size, lower.shape[0]) * (upper - lower) + lower,
            np.ones(size, dtype=float) / size)


# 2d gauss

def gauss_2d_iid(size: int, params: tuple = ((0.2, 0.1, 0.3, 0.2, 0.3),
                                             (0.7, 0.2, 0.6, 0.15, 0.7)),
                 region: tuple = ((0., 1.), (0., 1.))) -> tuple:
    """
    Gauss 2d sampler, iid.
    """

    spec = np.random.multinomial(size, [x[-1] for x in params])
    left_x, right_x = region[0]
    left_y, right_y = region[1]
    sample = np.zeros([size, 2], dtype=float)
    current = 0
    for i, size_i in enumerate(spec):
        loc, scale = params[i][:2]
        sample_x = st.truncnorm((left_x - loc) / scale,
                                (right_x - loc) / scale,
                                loc=loc, scale=scale).rvs(size_i)
        sample[current:current+size_i, 0] = sample_x
        loc, scale = params[i][2:4]
        sample_y = st.truncnorm((left_y - loc) / scale,
                                (right_y - loc) / scale,
                                loc=loc, scale=scale).rvs(size_i)
        sample[current:current+size_i, 1] = sample_y
        current += size_i

    return sample, np.ones(size, dtype=float) / size

def gauss_2d_even(size: tuple,
                  params: tuple = ((0.2, 0.1, 0.3, 0.2, 0.3),
                                   (0.7, 0.2, 0.6, 0.15, 0.7)),
                  region: tuple = ((0., 1.), (0., 1.))) -> tuple:
    """
    Gauss 2d sampler, evenly positioned
    """
    left_x, right_x = region[0]
    left_y, right_y = region[1]
    # ratio = int((right_y - left_y) / (right_x - left_x))
    if isinstance(size, int):
        density_1d = int((size) ** 0.5 + 0.5)
        size = [density_1d, density_1d]
    len_x = 0.5 * (right_x - left_x) / size[0]
    len_y = 0.5 * (right_y - left_y) / size[1]
    lattice_x = np.concatenate([np.linspace(left_x + len_x,
                                            right_x - len_x,
                                            size[0]).reshape(-1, 1, 1),
                                np.ones(size[0]).reshape(-1, 1, 1)],
                               axis=2)
    lattice_y = np.concatenate([np.ones(size[1],
                                        dtype=float).reshape(1, -1, 1),
                                np.linspace(left_y + len_y,
                                            right_y - len_y,
                                            size[1]).reshape(1, -1, 1)],
                               axis=2)
    lattice = (lattice_x * lattice_y).reshape(-1, 2)


    weight = np.zeros(size[0] * size[1], dtype=float)
    for param in params:
        loc, scale = param[:2]
        wt_x = st.truncnorm((left_x - loc) / scale,
                            (right_x - loc) / scale,
                            loc=loc, scale=scale).pdf(lattice[:, 0])
        wt_x *= size[1] / np.sum(wt_x)

        loc, scale = param[2:4]
        wt_y = st.truncnorm((left_y - loc) / scale,
                            (right_y - loc) / scale,
                            loc=loc, scale=scale).pdf(lattice[:, 1])
        wt_y *= size[0] / np.sum(wt_y)
        weight += wt_x * wt_y * param[4]

    return lattice, weight

# swiss_roll


def swiss_roll_transform(points: np.ndarray,
                         coeff: float = 1):
    """
    Swiss Roll Transform

    Use spiral to construct a map from 2d-coordinates
    into a 3d space
    """
    points = points.reshape(-1, 2)
    r = points[:, 0]
    z = points[:, 1]
    ret = np.zeros([z.shape[0], 3], dtype=float)
    ret[:, 0] = r * np.cos(np.pi * coeff * r)
    ret[:, 1] = r * np.sin(np.pi * coeff * r)
    ret[:, 2] = z
    return ret

def swiss_roll_general(size: int,
                       pos_sampler: callable,
                       r: tuple = (1., 4.),
                       z: tuple = (0., 1.),
                       coeff: float = 1.):
    """
    Sample uniformly on a spiral surface

    Parameters
    ----------
    pos_sampler: callable(size) -> (points, weights)


    Return
    ------
    """
    region_shape = np.array([r[1] - r[0], z[1] - z[0]])
    sample_2d, weight = pos_sampler(size)
    sample_2d = sample_2d * region_shape + np.array([r[0], z[0]])

    return swiss_roll_transform(sample_2d, coeff=coeff), weight


def swiss_roll_uniform_iid(size: int,
                           r: tuple = (1., 4.),
                           z: tuple = (0., 1.),
                           coeff: float = 1.) -> tuple:
    """
    Sample uniformly on a spiral surface
    """
    uniform_pos_sampler = lambda size: (np.random.rand(size, 2),
                                        np.ones(size, dtype=float) / size)
    return swiss_roll_general(size, uniform_pos_sampler,
                              r, z, coeff)

def swiss_roll_uniform_even(size: tuple,
                            r_range: tuple = (1., 4.),
                            z_range: tuple = (0., 1.),
                            coeff: float = 1.) -> tuple:
    """
    Swiss Roll sampler, evenly distributed
    """
    left_x, right_x = r_range
    left_y, right_y = z_range

    if isinstance(size, int):
        density_1d = int((size) ** 0.5 + 0.5)
        size = [density_1d, density_1d]
    lattice_x = np.concatenate([np.linspace(left_x,
                                            right_x,
                                            size[0]).reshape(-1, 1, 1),
                                np.ones(size[0]).reshape(-1, 1, 1)],
                               axis=2)
    lattice_y = np.concatenate([np.ones(size[1],
                                        dtype=float).reshape(1, -1, 1),
                                np.linspace(left_y,
                                            right_y,
                                            size[1]).reshape(1, -1, 1)],
                               axis=2)
    lattice = (lattice_x * lattice_y).reshape(-1, 2)

    return (swiss_roll_transform(lattice, coeff),
            np.ones(size[0] * size[1]) / size[0] * size[1])


def swiss_roll_gauss_even(size: tuple,
                          params: tuple = ((0.2, 0.1, 0.3, 0.2, 0.3),
                                           (0.7, 0.2, 0.6, 0.15, 0.7)),
                          r_range: tuple = (0., 5.),
                          z_range: tuple = (0., 1.),
                          coeff: float = 1.) -> tuple:
    """
    Gaussian distribution on Swiss Roll

    Parameters
    ----------
    size: tuple of int, the sizes.
    """
    sample_2d, weight = gauss_2d_even(size, params, region=(r_range, z_range))
    return swiss_roll_transform(sample_2d, coeff), weight

def swiss_roll_gauss_iid(size: int,
                         params: tuple = ((0.2, 0.1, 0.3, 0.2, 0.3),
                                          (0.7, 0.2, 0.6, 0.15, 0.7)),
                         r_range: tuple = (0., 5.),
                         z_range: tuple = (0., 1.),
                         coeff: float = 1.) -> tuple:
    """
    Gaussian Distribution on Swiss Roll, IID-version
    """
    sample_2d, weight = gauss_2d_iid(size, params, region=(r_range, z_range))
    return swiss_roll_transform(sample_2d, coeff), weight

# Distribution on a sphere

def transform_matrix(scale: np.ndarray,
                     theta: float) -> np.ndarray:
    """
    Generate a transformation matrix from scale and angle
    """
    return scale.reshape(-1, 1) * np.array([[np.cos(theta), np.sin(theta)],
                                            [-np.sin(theta), np.cos(theta)]])


def inv_stereo_proj(points: np.ndarray):
    """
    Inverse Stereographic Projection

    Parameters
    ----------
    points: numpy array of shape (n, 2)

    Return
    ------
    numpy array of shape (n, 3), the result
    """
    points = points.reshape(-1, 2)
    size = points.shape[0]
    result = np.zeros([size, 3], dtype=float)
    div = 1 + np.sum(points ** 2, axis=1)
    result[:, :2] = 2 * points / div.reshape(-1, 1)
    result[:, -1] = (div - 2) / div
    return result


def sphere_gauss_iid(size: int,
                     params: tuple = ((np.array([3, 1]),
                                       transform_matrix(np.array([1/2, 2]),
                                                        np.pi / 3),
                                       0.4),
                                      (np.array([0, -4]),
                                       transform_matrix(np.array([3, 1]),
                                                        -np.pi / 4),
                                       0.2),
                                      (np.array([-0.1, 0.2]),
                                       transform_matrix(np.array([0.2, 0.2]),
                                                        0.),
                                       0.4))):
    """
    Stereographic Gaussian Mixture: iid

    Sample from 2d gaussian mixture, and use
        stereographic projection to map samples onto a unit sphere.

    Parameters
    ----------
    size: int, number of samples
    params: tuple with elements of format (center, transform_matrix, weight),
            `transform_matrix` $M$ satisfies: $M^T.M$ is the covariance.

    Return
    ------
    np.ndarray of shape (size, 3), the samples on the unit sphere
    """
    spec = np.random.multinomial(size, [x[-1] for x in params])
    sample_blocks = []
    for i, size_i in enumerate(spec):
        sample_raw = np.random.normal(size=size_i * 2).reshape(size_i, 2)
        sample_blocks += [np.matmul(sample_raw, params[i][1]) + params[i][0], ]

    return (inv_stereo_proj(np.concatenate(sample_blocks, axis=0)),
            np.ones(size, dtype=float))
            # sample_blocks,)

def fibonacci_lattice_angles(size: int):
    """
    Fibonacci Lattice on Sphere
        in terms of latitude and longitude

    Parameters
    ----------
    size: int, should be an odd number

    Return
    ------
    latitude, longitude
    """
    half = size // 2
    phi = np.sqrt(5) / 2 + 0.5
    seq = np.arange(-half, half + 1)
    lat = np.arcsin(2 * seq / (2 * half + 1))
    lon = np.mod(seq, phi) * 2 * np.pi / phi
    return lat, lon

def fibonacci_lattice_stereo(size: int):
    """
    Fibonacci Lattice on Sphere
        in terms of stereographic projection images

    Use `inv_stereo_proj` to get 3d coordinates

    Parameters
    ----------
    size: int, should be an odd number

    Return
    ------
    points in terms of (x_coord, y_coord)
    """
    half = size // 2
    phi = np.sqrt(5) / 2 + 0.5
    seq = np.arange(-half, half + 1)
    radius = 2 * seq / (2 * half + 1)
    lon = np.mod(seq, phi) * 2 * np.pi / phi

    radius = np.sqrt((1 + radius) / (1 - radius))
    return np.array([radius * np.cos(lon),
                     radius * np.sin(lon)]).T


def sphere_gauss_even(size: int,
                      params: tuple = ((np.array([3, 1]),
                                        transform_matrix(np.array([1/2, 2]), np.pi / 3),
                                        0.4),
                                       (np.array([0, -4]),
                                        transform_matrix(np.array([3, 1]), -np.pi / 4),
                                        0.2),
                                       (np.array([-0.1, 0.2]),
                                        transform_matrix(np.array([0.2, 0.2]), 0.),
                                        0.4))):
    """
    Stereographic Gaussian Mixture: Evenly Distributed

    Sample from 2d gaussian mixture, and use
        stereographic projection to map samples onto a unit sphere.

    Parameters
    ----------
    size: int, number of samples
    params: tuple with elements of format (center, transform_matrix, weight),
            `transform_matrix` $M$ satisfies: $M^T.M$ is the covariance.

    Return
    ------
    np.ndarray of shape (size, 3), the samples on the unit sphere
    """

    positions = fibonacci_lattice_stereo(size)
    print(positions.shape)
    weight = np.zeros(positions.shape[0], dtype=float)
    for param in params:
        original = np.matmul(positions - param[0],
                             np.linalg.inv(param[1]))
        wt_sub = st.norm.pdf(original[:, 0]) * st.norm.pdf(original[:, 1])
        weight += wt_sub * param[2]

    return inv_stereo_proj(positions), weight

if __name__ == '__main__':
    pass
