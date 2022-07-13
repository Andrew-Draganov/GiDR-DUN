# cython: language_level=3

import numpy as py_np
cimport numpy as np
cimport cython
from libc.stdio cimport printf
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange, parallel
from sklearn.neighbors._quad_tree cimport _QuadTree
np.import_array()
INF = py_np.inf

cdef extern from "../utils/cython_utils.cpp" nogil:
    float clip(float value, float lower, float upper)
cdef extern from "../utils/cython_utils.cpp" nogil:
    int random_int(const int & max)
cdef extern from "../utils/cython_utils.cpp" nogil:
    float sq_euc_dist(float* x, float* y, int dim)
cdef extern from "../utils/cython_utils.cpp" nogil:
    float ang_dist(float* x, float* y, int dim)
cdef extern from "../utils/cython_utils.cpp" nogil:
    float get_lr(float initial_lr, int i_epoch, int n_epochs, int amplify_grads) 
cdef extern from "../utils/cython_utils.cpp" nogil:
    void print_status(int i_epoch, int n_epochs)
cdef extern from "../utils/cython_utils.cpp" nogil:
    float get_avg_weight(float* weights, int n_edges)
cdef extern from "../utils/cython_utils.cpp" nogil:
    float attractive_force_func(
            int normalized,
            int frob,
            float dist,
            float a,
            float b,
            float edge_weight
    )
cdef extern from "../utils/cython_utils.cpp" nogil:
    void repulsive_force_func(
            float* rep_func_outputs,
            int normalized,
            int frob,
            float dist,
            float a,
            float b,
            float cell_size,
            float average_weight
    )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _gdr_epoch(
    int normalized,
    int angular,
    int sym_attraction,
    int frob,
    int num_threads,
    int amplify_grads,
    float[:, :] embedding,
    int[:] head,
    int[:] tail,
    float[:] weights,
    float[:, :] attr_grads,
    float[:, :] rep_grads,
    float average_weight,
    float[:, :] all_updates,
    float[:, :] gains,
    float a,
    float b,
    int dim,
    int n_vertices,
    int negative_sample_rate,
    float lr,
    int i_epoch
):
    cdef:
        int i, j, k, v, d, edge
        float grad_d, weight_scalar, dist
        float attr_force, rep_force

    cdef int n_edges = int(head.shape[0])
    cdef float Z = 0

    # t-SNE early exaggeration factor
    if amplify_grads and i_epoch < 250:
        weight_scalar = 4
    else:
        weight_scalar = 1

    cdef float *y1
    cdef float *y2
    cdef float *rep_func_outputs
    with nogil, parallel(num_threads=num_threads):
        # Zero out per-edge forces
        for v in prange(n_vertices):
            for d in range(dim):
                attr_grads[v, d] = 0
                rep_grads[v, d] = 0

        # Initialize thread-local variables
        y1 = <float*> malloc(sizeof(float) * dim)
        y2 = <float*> malloc(sizeof(float) * dim)
        rep_func_outputs = <float*> malloc(sizeof(float) * 2)
        for edge in prange(n_edges):
            j = head[edge]
            k = tail[edge]
            for d in range(dim):
                y1[d] = embedding[j, d]
                y2[d] = embedding[k, d]

            if angular:
                dist = ang_dist(y1, y2, dim)
            else:
                dist = sq_euc_dist(y1, y2, dim)

            attr_force = attractive_force_func(
                normalized,
                frob,
                dist,
                a,
                b,
                weights[edge] * weight_scalar
            )
            for d in range(dim):
                grad_d = attr_force * (y1[d] - y2[d])
                attr_grads[j, d] -= grad_d
                if sym_attraction:
                    attr_grads[k, d] += grad_d

            k = random_int(n_vertices)
            for d in range(dim):
                y2[d] = embedding[k, d]

            if angular:
                dist = ang_dist(y1, y2, dim)
            else:
                dist = sq_euc_dist(y1, y2, dim)

            # Simplify this call like in numba optimizers
            repulsive_force_func(
                rep_func_outputs,
                normalized,
                frob,
                dist,
                a,
                b,
                1.0,
                average_weight,
            )
            rep_force = rep_func_outputs[0]
            Z += rep_func_outputs[1]

            for d in range(dim):
                grad_d = rep_force * (y1[d] - y2[d])
                rep_grads[j, d] += grad_d

        free(rep_func_outputs)
        free(y1)
        free(y2)

    if not normalized:
        Z = 1

    with nogil, parallel(num_threads=num_threads):
        for v in prange(n_vertices):
            for d in range(dim):
                grad_d = 4 * a * b * (attr_grads[v, d] + rep_grads[v, d] / Z)

                # Update gains variable, which acts like an additional momentum term
                if grad_d * all_updates[v, d] > 0.0:
                    gains[v, d] += 0.2
                else:
                    gains[v, d] *= 0.8
                gains[v, d] = clip(gains[v, d], 0.01, 1000)
                grad_d = clip(grad_d * gains[v, d], -1, 1)

                all_updates[v, d] = grad_d * lr + amplify_grads * 0.9 * all_updates[v, d]
                embedding[v, d] += all_updates[v, d]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef gdr_optimize(
    int normalized,
    int angular,
    int sym_attraction,
    int frob,
    int num_threads,
    int amplify_grads,
    float[:, :] embedding,
    int[:] head,
    int[:] tail,
    float[:] weights,
    float a,
    float b,
    int dim,
    float initial_lr,
    int n_epochs,
    int n_vertices,
    int negative_sample_rate,
    int verbose
):
    cdef:
        int v, d, edge

    # Initialize arrays used for gradient descent for-loop
    py_all_updates = py_np.zeros([n_vertices, dim], dtype=py_np.float32)
    py_gains = py_np.ones([n_vertices, dim], dtype=py_np.float32)
    cdef float[:, :] all_updates = py_all_updates
    cdef float[:, :] gains = py_gains

    # Initialize arrays used for calculating forces acting on each point
    cdef int n_edges = int(head.shape[0])
    py_attr_grads = py_np.zeros([n_edges, dim], dtype=py_np.float32)
    py_rep_grads = py_np.zeros([n_edges, dim], dtype=py_np.float32)
    cdef float average_weight = get_avg_weight(&weights[0], n_edges)
    cdef float[:, :] attr_grads = py_attr_grads
    cdef float[:, :] rep_grads = py_rep_grads

    for i_epoch in range(n_epochs):
        lr = get_lr(initial_lr, i_epoch, n_epochs, amplify_grads)
        _gdr_epoch(
            normalized,
            angular,
            sym_attraction,
            frob,
            num_threads,
            amplify_grads,
            embedding,
            head,
            tail,
            weights,
            attr_grads,
            rep_grads,
            average_weight,
            all_updates,
            gains,
            a,
            b,
            dim,
            n_vertices,
            negative_sample_rate,
            lr,
            i_epoch
        )
        # FIXME -- why is this still printing even for verbose=False
        if verbose:
            print_status(i_epoch, n_epochs)



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def gdr_opt_wrapper(
    int normalized,
    int angular,
    int sym_attraction,
    int frob,
    int num_threads,
    int amplify_grads,
    float[:, :] head_embedding,
    int[:] head,
    int[:] tail,
    float[:] weights,
    int n_epochs,
    int n_vertices,
    float a,
    float b,
    float initial_lr,
    int negative_sample_rate,
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

    gdr_optimize(
        normalized,
        angular,
        sym_attraction,
        frob,
        num_threads,
        amplify_grads,
        head_embedding,
        head,
        tail,
        weights,
        a,
        b,
        dim,
        initial_lr,
        n_epochs,
        n_vertices,
        negative_sample_rate,
        verbose
    )

    return py_np.asarray(head_embedding)
