# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 3 clause
import numpy as np
import numba

@numba.njit(fastmath=True)
def euclidean(x, y):
    r"""Standard euclidean distance.

    .. math::
        D(x, y) = \\sqrt{\sum_i (x_i - y_i)^2}
    """
    result = 0.0
    for i in range(x.shape[0]):
        # ANDREW - fastpow
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)
