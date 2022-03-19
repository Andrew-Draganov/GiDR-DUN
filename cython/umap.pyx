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
    float get_lr(float initial_lr, int i_epoch, int n_epochs, int amplify_grads) 
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
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _frob_umap_sampling(
    int normalized,
    int sym_attraction,
    int frob,
    int num_threads,
    int amplify_grads,
    float[:, :] head_embedding,
    float[:, :] tail_embedding,
    int[:] head,
    int[:] tail,
    float[:] weights,
    float* all_updates,
    float* gains,
    float a,
    float b,
    int dim,
    int n_vertices,
    float lr,
    float[:] epochs_per_sample,
    float[:] epochs_per_negative_sample,
    float[:] epoch_of_next_negative_sample,
    float[:] epoch_of_next_sample,
    int i_epoch,
):
    cdef:
        int i, j, k, n_neg_samples, edge, p
        int index1, index2
        # Can't reuse loop variables in a with nogil, parallel(): block
        int v1, v2
        int d1, d2, d3, d4, d5, d6
        float grad_d1, grad_d2, grad_d3, grad_d4
        float weight_scalar = 1
        float attractive_force, repulsive_force
        float dist_squared
        float *y1
        float *y2
        float *all_attr_grads
        float *all_rep_grads
        float *rep_func_outputs

    # FIXME FIXME -- normalized doesn't work here!
    cdef float Z = 0
    cdef int n_edges = int(head.shape[0])
    all_attr_grads = <float*> malloc(sizeof(float) * n_vertices * dim)
    all_rep_grads = <float*> malloc(sizeof(float) * n_vertices * dim)
    with nogil, parallel(num_threads=num_threads):
        rep_func_outputs = <float*> malloc(sizeof(float) * 2)
        y1 = <float*> malloc(sizeof(float) * dim)
        y2 = <float*> malloc(sizeof(float) * dim)
        for v1 in prange(n_vertices):
            for d1 in range(dim):
                index1 = v1 * dim + d1
                all_attr_grads[index1] = 0
                all_rep_grads[index1] = 0

        for edge in prange(n_edges):
            if epoch_of_next_sample[edge] <= i_epoch:
                # Gets one of the knn in HIGH-DIMENSIONAL SPACE relative to the sample point
                j = head[edge]
                k = tail[edge]

                for d2 in range(dim):
                    y1[d2] = head_embedding[j, d2]
                    y2[d2] = tail_embedding[k, d2]
                dist_squared = sq_euc_dist(y1, y2, dim)

                if amplify_grads and i_epoch < 250:
                    weight_scalar = 4
                else:
                    weight_scalar = 1
                attractive_force = attractive_force_func(
                    normalized,
                    frob,
                    dist_squared,
                    a,
                    b,
                    1.0 * weight_scalar
                )

                for d3 in range(dim):
                    grad_d1 = clip(attractive_force * (y1[d3] - y2[d3]), -4, 4)
                    all_attr_grads[j * dim + d3] -= grad_d1
                    if sym_attraction:
                        all_attr_grads[k * dim + d3] += grad_d1

                epoch_of_next_sample[edge] += epochs_per_sample[edge]
                n_neg_samples = int(
                    (i_epoch - epoch_of_next_negative_sample[edge]) / epochs_per_negative_sample[edge]
                )

                for p in range(n_neg_samples):
                    k = rand() % n_vertices
                    for d4 in range(dim):
                        y2[d4] = tail_embedding[k, d4]
                    dist_squared = sq_euc_dist(y1, y2, dim)
                    repulsive_force_func(
                        rep_func_outputs,
                        normalized,
                        frob,
                        dist_squared,
                        a,
                        b,
                        1.0,
                        0.0
                    )
                    repulsive_force = rep_func_outputs[0]
                    Z += rep_func_outputs[1]

                    for d5 in range(dim):
                        grad_d2 = clip(repulsive_force * (y1[d5] - y2[d5]), -4, 4)
                        all_rep_grads[j * dim + d5] += grad_d2

                epoch_of_next_negative_sample[edge] += (
                    n_neg_samples * epochs_per_negative_sample[edge]
                )

        free(y1)
        free(y2)
        free(rep_func_outputs)

    if not normalized or Z == 0:
        Z = 1

    with nogil, parallel(num_threads=num_threads):
        for v2 in prange(n_vertices):
            for d6 in range(dim):
                index2 = v2 * dim + d6
                grad_d3 = (all_rep_grads[index2] / Z + all_attr_grads[index2]) * gains[index2]

                head_embedding[v2, d6] += grad_d3 * lr

    free(all_attr_grads)
    free(all_rep_grads)


cdef umap_optimize(
    int normalized,
    int sym_attraction,
    int frob,
    int num_threads,
    int amplify_grads,
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
        int i
        np.ndarray[DTYPE_FLOAT, ndim=1] epochs_per_negative_sample
        np.ndarray[DTYPE_FLOAT, ndim=1] epoch_of_next_negative_sample
        np.ndarray[DTYPE_FLOAT, ndim=1] epoch_of_next_sample
        float *all_updates
        float *gains

    all_updates = <float*> malloc(sizeof(float) * n_vertices * dim)
    gains = <float*> malloc(sizeof(float) * n_vertices * dim)

    epochs_per_negative_sample = py_np.zeros_like(epochs_per_sample)
    epoch_of_next_negative_sample = py_np.zeros_like(epochs_per_sample)
    epoch_of_next_sample = py_np.zeros_like(epochs_per_sample)
    # ANDREW - perform negative samples x times more often
    #          by making the number of epochs between samples smaller
    for i in range(weights.shape[0]):
        epochs_per_negative_sample[i] = epochs_per_sample[i] / negative_sample_rate
        epoch_of_next_negative_sample[i] = epochs_per_negative_sample[i]
        epoch_of_next_sample[i] = epochs_per_sample[i]

    for v in range(n_vertices):
        for d in range(dim):
            all_updates[v * dim + d] = 0
            gains[v * dim + d] = 1

    for i_epoch in range(n_epochs):
        lr = get_lr(initial_lr, i_epoch, n_epochs, amplify_grads)
        _frob_umap_sampling(
            normalized,
            sym_attraction,
            frob,
            num_threads,
            amplify_grads,
            head_embedding,
            tail_embedding,
            head,
            tail,
            weights,
            all_updates,
            gains,
            a,
            b,
            dim,
            n_vertices,
            lr,
            epochs_per_sample,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            i_epoch,
        )
        if verbose:
            print_status(i_epoch, n_epochs)

    free(all_updates)
    free(gains)


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def umap_opt_wrapper(
    str optimize_method,
    int normalized,
    int sym_attraction,
    int frob,
    int num_threads,
    int amplify_grads,
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
        num_threads,
        amplify_grads,
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
