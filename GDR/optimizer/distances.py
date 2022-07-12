# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 3 clause
import warnings
import os
import numba
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def remove_diag(A):
    removed = A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],int(A.shape[0])-1, -1)
    return np.squeeze(removed)

@numba.njit(fastmath=True)
def euclidean(x, y):
    """Standard euclidean distance.

    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)


@numba.njit(fastmath=True)
def euclidean_grad(x, y):
    """Standard euclidean distance and its gradient.

    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
        \frac{dD(x, y)}{dx} = (x_i - y_i)/D(x,y)
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    d = np.sqrt(result)
    grad = (x - y) / (1e-6 + d)
    return d, grad
