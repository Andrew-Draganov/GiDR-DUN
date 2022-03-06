import numpy as py_np
cimport numpy as np
cimport cython
from libc.stdio cimport printf
from libc.math cimport sqrt, pow
from libc.stdlib cimport rand
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange, parallel

np.import_array()

cdef float ang_dist(float* x, float* y, int dim):
    """ cosine distance between vectors x and y """
    cdef float result = 0.0
    cdef float x_len  = 0.0
    cdef float y_len  = 0.0
    cdef float eps = 0.0001
    cdef int i = 0
    for i in range(dim):
        result += x[i] * y[i]
        x_len += x[i] * x[i]
        y_len += y[i] * y[i]
    if x_len < eps or y_len < eps:
        return 1
    return result / (sqrt(x_len * y_len))


