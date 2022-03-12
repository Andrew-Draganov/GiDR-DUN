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

cdef extern from "cython_utils.c" nogil:
    float clip(float value, float lower, float upper)
cdef extern from "cython_utils.c" nogil:
    float sq_euc_dist(float* x, float* y, int dim)
cdef extern from "cython_utils.c" nogil:
    float get_lr(float initial_lr, int i_epoch, int n_epochs) 
cdef extern from "cython_utils.c" nogil:
    void print_status(int i_epoch, int n_epochs)
cdef extern from "cython_utils.c" nogil:
    float umap_repulsion_grad(float dist_squared, float a, float b)
cdef extern from "cython_utils.c" nogil:
    float kernel_function(float dist_squared, float a, float b)
cdef extern from "cython_utils.c" nogil:
    float attractive_force_func(
            int normalized,
            int frob,
            float dist_squared,
            float a,
            float b,
            float edge_weight
    )
cdef extern from "cython_utils.c" nogil:
    void repulsive_force_func(
            float* rep_func_outputs,
            int normalized,
            int frob,
            float dist_squared,
            float a,
            float b,
            float cell_size,
            float average_weight
    )

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

ctypedef np.float32_t DTYPE_FLOAT
ctypedef np.int32_t DTYPE_INT

@cython.boundscheck(False)
@cython.cdivision(True)
cdef float get_avg_weight(float[:] weights) nogil:
    cdef float average_weight = 0.0
    for i in range(weights.shape[0]):
        average_weight += weights[i]
    average_weight /= float(weights.shape[0])
    return average_weight

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _umap_epoch(
    int normalized,
    int sym_attraction,
    int frob,
    float[:, :] head_embedding,
    float[:, :] tail_embedding,
    int[:] head,
    int[:] tail,
    float[:] weights,
    float[:] epochs_per_sample,
    float a,
    float b,
    int dim,
    int n_vertices,
    float lr,
    float[:] epochs_per_negative_sample,
    float[:] epoch_of_next_negative_sample,
    float[:] epoch_of_next_sample,
    int i_epoch,
):
    cdef:
        int i, j, k, n_neg_samples, edge, p
        # Can't reuse loop variables in a with nogil, parallel(): block,
        #   so we create a new variable for each loop
        int v1, v2
        int d1, d2, d3, d4
        float grad_d1, grad_d2
        float attractive_force, repulsive_force
        float dist_squared
        float *y1
        float *y2
        float *rep_func_outputs

    cdef float grad_d = 0.0
    cdef int n_edges = int(epochs_per_sample.shape[0])

    with nogil, parallel():
        y1 = <float*> malloc(sizeof(float) * dim)
        y2 = <float*> malloc(sizeof(float) * dim)
        rep_func_outputs = <float*> malloc(sizeof(float) * 2)
        for i in prange(epochs_per_sample.shape[0]):
            if epoch_of_next_sample[i] <= i_epoch:
                # Gets one of the knn in HIGH-DIMENSIONAL SPACE relative to the sample point
                j = head[i]
                k = tail[i]

                for d1 in range(dim):
                    y1[d1] = head_embedding[j, d1]
                    y2[d1] = tail_embedding[k, d1]
                dist_squared = sq_euc_dist(y1, y2, dim)
                attractive_force = attractive_force_func(
                    normalized,
                    frob,
                    dist_squared,
                    a,
                    b,
                    1
                )

                for d2 in range(dim):
                    grad_d1 = clip(attractive_force * (y1[d2] - y2[d2]), -4, 4)
                    head_embedding[j, d2] -= grad_d1 * lr
                    if sym_attraction:
                        head_embedding[k, d2] += grad_d1 * lr

                epoch_of_next_sample[i] += epochs_per_sample[i]

                # ANDREW - Picks random vertices from ENTIRE graph and calculates repulsive forces
                # FIXME - add random seed option
                n_neg_samples = int(
                    (i_epoch - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
                )
                for p in range(n_neg_samples):
                    k = rand() % n_vertices
                    for d3 in range(dim):
                        y2[d3] = tail_embedding[k, d3]
                    dist_squared = sq_euc_dist(y1, y2, dim)
                    repulsive_force_func(
                        rep_func_outputs,
                        normalized,
                        frob,
                        dist_squared,
                        a,
                        b,
                        1.0,
                        0, # Don't scale by weight since we sample according to weight distribution
                    )
                    repulsive_force = rep_func_outputs[0]

                    for d4 in range(dim):
                        grad_d2 = clip(repulsive_force * (y1[d4] - y2[d4]), -4, 4)
                        head_embedding[j, d4] += grad_d2 * lr

                epoch_of_next_negative_sample[i] += (
                    n_neg_samples * epochs_per_negative_sample[i]
                )
        free(y1)
        free(y2)

cdef umap_optimize(
    int normalized,
    int sym_attraction,
    int frob,
    int momentum,
    float[:, :] head_embedding,
    float[:, :] tail_embedding,
    int[:] head,
    int[:] tail,
    float[:] weights,
    float[:] epochs_per_sample,
    float a,
    float b,
    int dim,
    float initial_lr,
    float negative_sample_rate,
    int n_epochs,
    int n_vertices,
    int verbose
):
    cdef:
        np.ndarray[DTYPE_FLOAT, ndim=1] epochs_per_negative_sample,
        np.ndarray[DTYPE_FLOAT, ndim=1] epoch_of_next_negative_sample,
        np.ndarray[DTYPE_FLOAT, ndim=1] epoch_of_next_sample,

    epochs_per_negative_sample = py_np.zeros_like(epochs_per_sample)
    epoch_of_next_negative_sample = py_np.zeros_like(epochs_per_sample)
    epoch_of_next_sample = py_np.zeros_like(epochs_per_sample)
    # ANDREW - perform negative samples x times more often
    #          by making the number of epochs between samples smaller
    for i in range(weights.shape[0]):
        epochs_per_negative_sample[i] = epochs_per_sample[i] / negative_sample_rate
        epoch_of_next_negative_sample[i] = epochs_per_negative_sample[i]
        epoch_of_next_sample[i] = epochs_per_sample[i]

    for i_epoch in range(n_epochs):
        lr = get_lr(initial_lr, i_epoch, n_epochs)
        _umap_epoch(
            normalized,
            sym_attraction,
            frob,
            head_embedding,
            tail_embedding,
            head,
            tail,
            weights,
            epochs_per_sample,
            a,
            b,
            dim,
            n_vertices,
            lr,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            i_epoch,
        )
        if verbose:
            print_status(i_epoch, n_epochs)


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def umap_opt_wrapper(
    str optimize_method,
    int normalized,
    int sym_attraction,
    int frob,
    int momentum,
    float[:, :] head_embedding,
    float[:, :] tail_embedding,
    int[:] head,
    int[:] tail,
    float[:] weights,
    int n_epochs,
    int n_vertices,
    np.ndarray[DTYPE_FLOAT, ndim=1] epochs_per_sample,
    float a,
    float b,
    float initial_lr,
    float negative_sample_rate,
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

    umap_optimize(
        normalized,
        sym_attraction,
        frob,
        momentum,
        head_embedding,
        tail_embedding,
        head,
        tail,
        weights,
        epochs_per_sample,
        a,
        b,
        dim,
        initial_lr,
        negative_sample_rate,
        n_epochs,
        n_vertices,
        verbose
    )

    return py_np.asarray(head_embedding)
