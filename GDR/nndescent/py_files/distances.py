# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 3 clause
import numpy as np
import numba

@numba.njit(fastmath=True)
def euclidean(x, y):
    result = 0.0
    for i in range(x.shape[0]):
        # ANDREW - fastpow
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)

@numba.njit(fastmath=True)
def cosine(x, y):
    result = 0.0
    norm_x = 0.0
    norm_y = 0.0
    for i in range(x.shape[0]):
        result += x[i] * y[i]
        norm_x += x[i] ** 2
        norm_y += y[i] ** 2

    if norm_x == 0.0 and norm_y == 0.0:
        return 0.0
    elif norm_x == 0.0 or norm_y == 0.0:
        return 1.0
    else:
        # FIXME -- why (1 - ...)
        return 1.0 - (result / np.sqrt(norm_x * norm_y))

