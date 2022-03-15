import numpy as py_np
cimport numpy as np
cimport cython
from libc.stdio cimport printf
from libc.math cimport sqrt
from libc.stdlib cimport rand
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange, parallel

from sklearn.neighbors._quad_tree cimport _QuadTree
np.import_array()

cdef extern from "gpu_dim_reduction.cpp":
    void gpu_umap_wrap(
        int normalized,
        int sym_attraction,
        int momentum,
        float* head_embedding,
        float* tail_embedding,
        int* head,
        int* tail,
        float* weights,
        long* neighbor_counts,
        float* all_updates,
        float* gains,
        float a,
        float b,
        int dim,
        int n_vertices,
        float initial_lr,
        int n_edges,
        int n_epochs
    )

ctypedef np.float32_t DTYPE_FLOAT
ctypedef np.int32_t DTYPE_INT

cdef uniform_umap_optimize(
    int normalized,
    int sym_attraction,
    int frob,
    int momentum,
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] head_embedding,
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] tail_embedding,
    np.ndarray[int, ndim=1, mode='c'] head,
    np.ndarray[int, ndim=1, mode='c'] tail,
    np.ndarray[DTYPE_FLOAT, ndim=1, mode='c'] weights,
    np.ndarray[long, ndim=1, mode='c'] neighbor_counts,
    float a,
    float b,
    int dim,
    float initial_lr,
    int n_epochs,
    int n_vertices,
    int verbose
):
    cdef:
        int v, d, index
        np.ndarray[DTYPE_FLOAT, ndim=2] all_updates
        np.ndarray[DTYPE_FLOAT, ndim=2] gains

    all_updates = py_np.zeros([n_vertices, dim], dtype=py_np.float32, order='c')
    gains = py_np.ones([n_vertices, dim], dtype=py_np.float32, order='c')

    cdef int n_edges = int(weights.shape[0])
    gpu_umap_wrap(
        normalized,
        sym_attraction,
        momentum,
        &head_embedding[0, 0], # Move from numpy to c pointer arrays
        &tail_embedding[0, 0],
        &head[0],
        &tail[0],
        &weights[0],
        &neighbor_counts[0],
        &all_updates[0, 0],
        &gains[0, 0],
        a,
        b,
        dim,
        n_vertices,
        initial_lr,
        n_edges,
        n_epochs
    )


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def uniform_umap_opt_wrapper(
    str optimize_method,
    int normalized,
    int sym_attraction,
    int frob,
    int momentum,
    np.ndarray[DTYPE_INT, ndim=1, mode='c'] head,
    np.ndarray[DTYPE_INT, ndim=1, mode='c'] tail,
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] head_embedding,
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] tail_embedding,
    np.ndarray[DTYPE_FLOAT, ndim=1, mode='c'] weights,
    np.ndarray[long, ndim=1, mode='c'] neighbor_counts,
    int n_epochs,
    int n_vertices,
    float a,
    float b,
    float initial_lr,
    int verbose=True,
    **kwargs
):
    cdef:
        int i_epoch, n_edges
        cdef int dim = head_embedding.shape[1]

    # Perform weight scaling on high-dimensional relationships
    cdef float weight_sum = 0.0
    if normalized:
        for i in range(weights.shape[0]):
            weight_sum += weights[i]
        for i in range(weights.shape[0]):
            weights[i] /= weight_sum
        initial_lr *= 200

    uniform_umap_optimize(
        normalized,
        sym_attraction,
        frob,
        momentum,
        head_embedding,
        tail_embedding,
        head,
        tail,
        weights,
        neighbor_counts,
        a,
        b,
        dim,
        initial_lr,
        n_epochs,
        n_vertices,
        verbose
    )

    return py_np.asarray(head_embedding)