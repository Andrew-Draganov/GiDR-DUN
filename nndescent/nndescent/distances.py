# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 3 clause
import numpy as np
import numba

from pynndescent.optimal_transport import (
    allocate_graph_structures,
    initialize_graph_structures,
    initialize_supply,
    initialize_cost,
    network_simplex_core,
    total_cost,
    ProblemStatus,
    sinkhorn_transport_plan,
)

_mock_identity = np.eye(2, dtype=np.float32)
_mock_ones = np.ones(2, dtype=np.float32)
_dummy_cost = np.zeros((2, 2), dtype=np.float64)

FLOAT32_EPS = np.finfo(np.float32).eps
FLOAT32_MAX = np.finfo(np.float32).max


@numba.njit(fastmath=True)
def euclidean(x, y):
    r"""Standard euclidean distance.

    .. math::
        D(x, y) = \\sqrt{\sum_i (x_i - y_i)^2}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)


@numba.njit(
    [
        "f4(f4[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "diff": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
)
def squared_euclidean(x, y):
    r"""Squared euclidean distance.

    .. math::
        D(x, y) = \sum_i (x_i - y_i)^2
    """
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result


@numba.njit(fastmath=True)
def standardised_euclidean(x, y, sigma=_mock_ones):
    r"""Euclidean distance standardised against a vector of standard
    deviations per coordinate.

    .. math::
        D(x, y) = \sqrt{\sum_i \frac{(x_i - y_i)**2}{v_i}}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += ((x[i] - y[i]) ** 2) / sigma[i]

    return np.sqrt(result)


named_distances = {
    # general minkowski distances
    "euclidean": euclidean,
    "l2": euclidean,
    "sqeuclidean": squared_euclidean,
}
