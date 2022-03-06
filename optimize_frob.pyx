import numpy as py_np
cimport numpy as np
cimport cython
from libc.stdio cimport printf, fflush
from libc.math cimport sqrt
from libc.stdlib cimport rand
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange, parallel, threadid
from cython.view cimport array as cvarray

np.import_array()

cdef extern from "fastpow.c" nogil:
    float clip "clip" (float, float, float)
cdef extern from "fastpow.c" nogil:
    float sq_euc_dist "sq_euc_dist" (float*, float*, int)
cdef extern from "fastpow.c" nogil:
    float get_lr "get_lr" (float, int, int)
cdef extern from "fastpow.c" nogil:
    void print_status "print_status" (int, int)
cdef extern from "fastpow.c" nogil:
    float umap_repulsion_grad "umap_repulsion_grad" (float, float, float)
cdef extern from "fastpow.c" nogil:
    float kernel_function "kernel_function" (float, float, float)
cdef extern from "fastpow.c" nogil:
    float pos_force "pos_force" (int, float, float, float)
cdef extern from "fastpow.c" nogil:
    float neg_force "neg_force" (int, float, float)

ctypedef np.float32_t DTYPE_FLOAT
ctypedef np.int32_t DTYPE_INT

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _frob_umap_sampling(
    int normalized,
    int sym_attraction,
    int momentum,
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
    int* pos_sample_steps,
    float negative_sample_rate,
    int i_epoch,
    int n_epochs
):
    cdef:
        int i, j, k, n_neg_samples, edge, p
        int index1, index2
        # Can't reuse loop variables in a with nogil, parallel(): block
        int v1, v2
        int d1, d2, d3, d4, d5, d6
        float grad_d1, grad_d2, grad_d3, grad_d4
        float align
        float attractive_force, repulsive_force
        float dist_squared
        float *y1
        float *y2
        float *all_attr_grads
        float *all_rep_grads

    if normalized:
        raise ValueError("Cannot perform frobenius umap sampling with normalization")

    cdef int n_edges = int(head.shape[0])
    with nogil, parallel():
        y1 = <float*> malloc(sizeof(float) * dim)
        y2 = <float*> malloc(sizeof(float) * dim)
        all_attr_grads = <float*> malloc(sizeof(float) * n_vertices * dim)
        all_rep_grads = <float*> malloc(sizeof(float) * n_vertices * dim)
        for v1 in prange(n_vertices):
            for d1 in range(dim):
                index1 = v1 * dim + d1
                all_attr_grads[index1] = 0
                all_rep_grads[index1] = 0

        for edge in prange(n_edges):
            if pos_sample_steps[edge * n_epochs + i_epoch]:
                # Gets one of the knn in HIGH-DIMENSIONAL SPACE relative to the sample point
                j = head[edge]
                k = tail[edge]

                for d2 in range(dim):
                    y1[d2] = head_embedding[j, d2]
                    y2[d2] = tail_embedding[k, d2]
                dist_squared = sq_euc_dist(y1, y2, dim)
                attractive_force = pos_force(
                    normalized,
                    weights[edge],
                    kernel_function(dist_squared, a, b),
                    1
                )

                for d3 in range(dim):
                    grad_d1 = attractive_force * (y1[d3] - y2[d3])
                    # Attractive force has a negative scalar on it
                    #   in frobenius norm gradient
                    all_attr_grads[j * dim + d3] -= grad_d1 * lr
                    if sym_attraction:
                        all_attr_grads[k * dim + d3] += grad_d1 * lr

                for p in range(int(negative_sample_rate)):
                    k = rand() % n_vertices
                    for d4 in range(dim):
                        y2[d4] = tail_embedding[k, d4]
                    dist_squared = sq_euc_dist(y1, y2, dim)
                    repulsive_force = neg_force(
                        normalized,
                        kernel_function(dist_squared, a, b),
                        1
                    )

                    for d5 in range(dim):
                        grad_d2 = repulsive_force * (y1[d5] - y2[d5])
                        all_rep_grads[j * dim + d5] += grad_d2 * lr
        free(y1)
        free(y2)

        for v2 in prange(n_vertices):
            for d6 in range(dim):
                index2 = v2 * dim + d6
                grad_d3 = all_attr_grads[index2] + all_rep_grads[index2]

                if grad_d3 * all_updates[index2] > 0.0:
                    gains[index2] += 0.2
                else:
                    gains[index2] *= 0.8
                grad_d4 = grad_d3 * gains[index2]
                all_updates[index2] = momentum * all_updates[index2] * 0.3 + grad_d4 * lr
                head_embedding[v2, d6] += all_updates[index2]

        free(all_attr_grads)
        free(all_rep_grads)


def frob_umap_sampling(
    int normalized,
    int sym_attraction,
    int momentum,
    float[:, :] head_embedding,
    float[:, :] tail_embedding,
    int[:] head,
    int[:] tail,
    float[:] weights,
    np.ndarray[DTYPE_FLOAT, ndim=1] epochs_per_sample,
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
        int v, d
        int* pos_sample_steps
        int i_edge, i_epoch
        int sample_rate, sample_bool, count
        int n_edges
        float lr
        float* all_updates,
        float* gains,

    n_edges = int(epochs_per_sample.shape[0])
    all_updates = <float*> malloc(sizeof(float) * n_vertices * dim)
    gains = <float*> malloc(sizeof(float) * n_vertices * dim)
    pos_sample_steps = <int*> malloc(sizeof(int) * n_edges * n_epochs)

    for i_edge in range(n_edges):
        sample_rate = int(epochs_per_sample[i_edge])
        count = 0
        for i_epoch in range(n_epochs):
            if count == sample_rate:
                sample_bool = 1
                count = -1
            else:
                sample_bool = 0

            pos_sample_steps[i_edge * n_epochs + i_epoch] = sample_bool
            count += 1

    for v in range(n_vertices):
        for d in range(dim):
            all_updates[v * dim + d] = 0
            gains[v * dim + d] = 1

    for i_epoch in range(n_epochs):
        lr = get_lr(initial_lr, i_epoch, n_epochs)
        _frob_umap_sampling(
            normalized,
            sym_attraction,
            momentum,
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
            pos_sample_steps,
            negative_sample_rate,
            i_epoch,
            n_epochs
        )
        if verbose:
            print_status(i_epoch, n_epochs)

    free(all_updates)
    free(gains)
    free(pos_sample_steps)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void get_kernels(
    float *attr_forces,
    float *rep_forces,
    float *attr_vecs,
    float *rep_vecs,
    int[:] head,
    int[:] tail,
    float[:, :] head_embedding,
    float[:, :] tail_embedding,
    float[:] weights,
    int normalized,
    int n_vertices,
    int n_edges,
    int dim,
    float a,
    float b
) nogil:
    cdef:
        int edge, j, k, d
        float dist_squared
        float *y1
        float *y2

    with parallel():
        y1 = <float*> malloc(sizeof(float) * dim)
        y2 = <float*> malloc(sizeof(float) * dim)
        for edge in prange(n_edges):
            j = head[edge]
            k = tail[edge]
            for d in range(dim):
                y1[d] = head_embedding[j, d]
                y2[d] = tail_embedding[k, d]
                attr_vecs[edge * dim + d] = y1[d] - y2[d]
            dist_squared = sq_euc_dist(y1, y2, dim)
            attr_forces[edge] = pos_force(
                normalized,
                weights[edge],
                kernel_function(dist_squared, a, b),
                0.0
            )

            k = rand() % n_vertices
            for d in range(dim):
                y2[d] = tail_embedding[k, d]
                rep_vecs[edge * dim + d] = y1[d] - y2[d]
            dist_squared = sq_euc_dist(y1, y2, dim)
            rep_forces[edge] = neg_force(
                normalized,
                kernel_function(dist_squared, a, b),
                0.0
            )

        free(y1)
        free(y2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void gather_gradients(
    float *local_attr_grads,
    float *local_rep_grads,
    int[:] head,
    int[:] tail,
    float* attr_forces,
    float* rep_forces,
    float* attr_vecs,
    float* rep_vecs,
    int sym_attraction,
    int n_vertices,
    int n_edges,
    int dim,
) nogil:
    # FIXME FIXME - Don't have Z in this version
    cdef:
        int j, k, v, d, edge, index
        float force, grad_d

    with parallel():
        for v in prange(n_vertices):
            for d in range(dim):
                index = v * dim + d
                local_attr_grads[index] = 0
                local_rep_grads[index] = 0

        for edge in prange(n_edges):
            j = head[edge]
            for d in range(dim):
                grad_d = attr_forces[edge] * attr_vecs[edge * dim + d]
                local_attr_grads[j * dim + d] += grad_d
                if sym_attraction:
                    k = tail[edge]
                    local_attr_grads[k * dim + d] -= grad_d

            for d in range(dim):
                grad_d = rep_forces[edge] * rep_vecs[edge * dim + d]
                local_rep_grads[j * dim + d] += grad_d


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _frob_umap_uniformly(
    int normalized,
    int sym_attraction,
    int momentum,
    float[:, :] head_embedding,
    float[:, :] tail_embedding,
    int[:] head,
    int[:] tail,
    float[:] weights,
    float* gains,
    float a,
    float b,
    int dim,
    int n_vertices,
    float lr,
    int i_epoch
):
    cdef:
        int i, j, k, v, d, index, edge 
        float dist_squared, Z
        float *local_attr_grads
        float *local_rep_grads
        float *all_attr_grads
        float *all_rep_grads
        float *attr_vecs
        float *rep_vecs
        float *attr_forces
        float *rep_forces
        float *all_updates

    cdef int n_edges = int(head.shape[0])
    all_attr_grads = <float*> malloc(sizeof(float) * n_vertices * dim)
    all_rep_grads = <float*> malloc(sizeof(float) * n_vertices * dim)
    # FIXME FIXME - momentum gradient descent! Define forces outside main loop
    all_updates = <float*> malloc(sizeof(float) * n_vertices * dim)
    attr_forces = <float*> malloc(sizeof(float) * n_edges)
    rep_forces = <float*> malloc(sizeof(float) * n_edges)
    attr_vecs = <float*> malloc(sizeof(float) * n_edges * dim)
    rep_vecs = <float*> malloc(sizeof(float) * n_edges * dim)
    local_attr_grads = <float*> malloc(sizeof(float) * n_vertices * dim)
    local_rep_grads = <float*> malloc(sizeof(float) * n_vertices * dim)
    with nogil:
        for v in prange(n_vertices):
            for d in range(dim):
                index = v * dim + d
                all_attr_grads[index] = 0
                all_rep_grads[index] = 0

    cdef float grad = 0.0
    cdef float grad_d = 0.0
    with nogil:
        get_kernels(
            attr_forces,
            rep_forces,
            attr_vecs,
            rep_vecs,
            head,
            tail,
            head_embedding,
            tail_embedding,
            weights,
            normalized,
            n_vertices,
            n_edges,
            dim,
            a,
            b
        )

        gather_gradients(
            local_attr_grads,
            local_rep_grads,
            head,
            tail,
            attr_forces,
            rep_forces,
            attr_vecs,
            rep_vecs,
            sym_attraction,
            n_vertices,
            n_edges,
            dim,
        )
        free(attr_forces)
        free(rep_forces)
        free(attr_vecs)
        free(rep_vecs)

        # Need to collect thread-local forces into single gradient
        #   before performing gradient descent
        with parallel():
            for v in prange(n_vertices):
                for d in range(dim):
                    index = v * dim + d
                    all_attr_grads[index] += local_attr_grads[index]
                    all_rep_grads[index] += local_rep_grads[index]

        with parallel():
            for v in prange(n_vertices):
                for d in range(dim):
                    index = v * dim + d
                    grad_d = all_rep_grads[index] - all_attr_grads[index]
                    head_embedding[v, d] += grad_d * lr
        free(local_attr_grads)
        free(local_rep_grads)

    free(all_updates)
    free(all_attr_grads)
    free(all_rep_grads)



def frob_umap_uniformly(
    int normalized,
    int sym_attraction,
    int momentum,
    float[:, :] head_embedding,
    float[:, :] tail_embedding,
    int[:] head,
    int[:] tail,
    float[:] weights,
    np.ndarray[DTYPE_FLOAT, ndim=1] epochs_per_sample,
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
        float* all_updates,
        float* gains,
    all_updates = <float*> malloc(sizeof(float) * n_vertices * dim)
    gains = <float*> malloc(sizeof(float) * n_vertices * dim)

    for i_epoch in range(n_epochs):
        lr = get_lr(initial_lr, i_epoch, n_epochs)
        _frob_umap_uniformly(
            normalized,
            sym_attraction,
            momentum,
            head_embedding,
            tail_embedding,
            head,
            tail,
            weights,
            gains,
            a,
            b,
            dim,
            n_vertices,
            lr,
            i_epoch
        )
        if verbose:
            print_status(i_epoch, n_epochs)

    free(all_updates)
    free(gains)


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def cy_optimize_frob(
    str optimize_method,
    int normalized,
    int sym_attraction,
    int momentum,
    np.ndarray[DTYPE_FLOAT, ndim=2] head_embedding,
    np.ndarray[DTYPE_FLOAT, ndim=2] tail_embedding,
    np.ndarray[DTYPE_INT, ndim=1] head,
    np.ndarray[DTYPE_INT, ndim=1] tail,
    np.ndarray[DTYPE_FLOAT, ndim=1] weights,
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
        int dim, i_epoch, v, d
        int n_edges

    dim = head_embedding.shape[1]
    cdef float [:, :] _head_embedding = head_embedding
    cdef float [:, :] _tail_embedding = tail_embedding
    cdef float [:] _weights = weights
    cdef int [:] _head = head
    cdef int [:] _tail = tail
    # FIXME FIXME FIXME - revisit normalized version here

    optimize_dict = {
        'frob_umap_uniform': frob_umap_uniformly,
        'frob_umap_sampling': frob_umap_sampling,
    }
    optimize_fn = optimize_dict[optimize_method]

    optimize_fn(
        normalized,
        sym_attraction,
        momentum,
        _head_embedding,
        _tail_embedding,
        _head,
        _tail,
        _weights,
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

    return head_embedding

