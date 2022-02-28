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
    double fastpow "fastPow" (double, double)

ctypedef np.float32_t DTYPE_FLOAT
ctypedef np.int32_t DTYPE_INT

cdef float sq_euc_dist(float* x, float* y, int dim) nogil:
    """ squared euclidean distance between x and y """
    cdef float result = 0.0
    cdef float diff = 0
    cdef int i
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result

cdef float get_lr(float initial_lr, int i_epoch, int n_epochs): 
    return initial_lr * (1.0 - (float(i_epoch) / float(n_epochs)))

cdef void print_status(int i_epoch, int n_epochs):
    cdef int print_rate = n_epochs / 10
    cdef int counter = 0
    # Can't do python modulo in cython
    while counter < i_epoch:
        counter += print_rate
    if i_epoch - counter == 0:
        printf("Completed %d / %d epochs\n", i_epoch, n_epochs)

@cython.cdivision(True)
cdef float kernel_function(float dist_squared, float a, float b) nogil:
    if b <= 1:
        return 1 / (1 + a * fastpow(dist_squared, b))
    return fastpow(dist_squared, b - 1) / (1 + a * fastpow(dist_squared, b))

cdef float pos_force(
    int normalized,
    float p,
    float q,
    float Z
) nogil:
    if normalized:
        # FIXME - is it faster to get q^2 and then use that for q^3?
        return Z * p * (q ** 2 + 2 * q ** 3)
    return p * q ** 2

cdef float neg_force(
    int normalized,
    float q,
    float Z
) nogil:
    if normalized:
        return Z * (q ** 3 + 2 * q ** 4)
    return q ** 3

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
    float* forces,
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
        float *attractive_forces
        float *repulsive_forces

    if normalized:
        raise ValueError("Cannot perform frobenius umap sampling with normalization")

    cdef int n_edges = int(head.shape[0])
    with nogil, parallel():
        y1 = <float*> malloc(sizeof(float) * dim)
        y2 = <float*> malloc(sizeof(float) * dim)
        attractive_forces = <float*> malloc(sizeof(float) * n_vertices * dim)
        repulsive_forces = <float*> malloc(sizeof(float) * n_vertices * dim)
        for v1 in prange(n_vertices):
            for d1 in range(dim):
                index1 = v1 * dim + d1
                attractive_forces[index1] = 0
                repulsive_forces[index1] = 0

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
                    Z=1
                )

                for d3 in range(dim):
                    grad_d1 = attractive_force * (y1[d3] - y2[d3])
                    # Attractive force has a negative scalar on it
                    #   in frobenius norm gradient
                    attractive_forces[j * dim + d3] -= grad_d1 * lr
                    if sym_attraction:
                        attractive_forces[k * dim + d3] += grad_d1 * lr

                for p in range(int(negative_sample_rate)):
                    k = rand() % n_vertices
                    for d4 in range(dim):
                        y2[d4] = tail_embedding[k, d4]
                    dist_squared = sq_euc_dist(y1, y2, dim)
                    repulsive_force = neg_force(
                        normalized,
                        kernel_function(dist_squared, a, b),
                        Z=1
                    )

                    for d5 in range(dim):
                        grad_d2 = repulsive_force * (y1[d5] - y2[d5])
                        repulsive_forces[j * dim + d5] += grad_d2 * lr
        free(y1)
        free(y2)

        for v2 in prange(n_vertices):
            for d6 in range(dim):
                index2 = v2 * dim + d6
                grad_d3 = attractive_forces[index2] + repulsive_forces[index2]

                if grad_d3 * forces[index2] > 0.0:
                    gains[index2] += 0.2
                else:
                    gains[index2] *= 0.8
                grad_d4 = grad_d3 * gains[index2]
                forces[index2] = momentum * forces[index2] * 0.3 + grad_d4 * lr
                head_embedding[v2, d6] += forces[index2]

        free(attractive_forces)
        free(repulsive_forces)


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
        float* forces,
        float* gains,

    n_edges = int(epochs_per_sample.shape[0])
    forces = <float*> malloc(sizeof(float) * n_vertices * dim)
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
            forces[v * dim + d] = 0
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
            forces,
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

    free(forces)
    free(gains)
    free(pos_sample_steps)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void get_kernels(
    float *Q1,
    float *Q2,
    float *attr_vecs,
    float *rep_vecs,
    int[:] head,
    int[:] tail,
    float[:, :] head_embedding,
    float[:, :] tail_embedding,
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
            Q1[edge] = kernel_function(dist_squared, a, b)

            k = rand() % n_vertices
            for d in range(dim):
                y2[d] = tail_embedding[k, d]
                rep_vecs[edge * dim + d] = y1[d] - y2[d]
            dist_squared = sq_euc_dist(y1, y2, dim)
            Q2[edge] = kernel_function(dist_squared, a, b)

        free(y1)
        free(y2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void calc_forces(
    float *local_attr_forces,
    float *local_rep_forces,
    int[:] head,
    int[:] tail,
    float[:] weights,
    float* Q1,
    float* Q2,
    float* attr_vecs,
    float* rep_vecs,
    int normalized,
    int sym_attraction,
    int n_vertices,
    int n_edges,
    int dim,
) nogil:
    cdef:
        int j, k, v, d, edge, index
        float grad, grad_d

    with parallel():
        for v in prange(n_vertices):
            for d in range(dim):
                index = v * dim + d
                local_attr_forces[index] = 0
                local_rep_forces[index] = 0

        for edge in prange(n_edges):
            j = head[edge]
            k = tail[edge]
            grad = pos_force(normalized, weights[edge], Q1[edge], 0.0)
            for d in range(dim):
                grad_d = grad * attr_vecs[edge * dim + d]
                local_attr_forces[j * dim + d] += grad_d
                if sym_attraction:
                    local_attr_forces[k * dim + d] -= grad_d

            grad = neg_force(normalized, Q2[edge], 0.0)
            for d in range(dim):
                local_rep_forces[j * dim + d] += grad * rep_vecs[edge * dim + d]


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
        float *local_attr_forces
        float *local_rep_forces
        float *attractive_forces
        float *repulsive_forces
        float *attr_vecs
        float *rep_vecs
        float *Q1
        float *Q2
        float *forces

    cdef int n_edges = int(head.shape[0])
    attractive_forces = <float*> malloc(sizeof(float) * n_vertices * dim)
    repulsive_forces = <float*> malloc(sizeof(float) * n_vertices * dim)
    forces = <float*> malloc(sizeof(float) * n_vertices * dim)
    Q1 = <float*> malloc(sizeof(float) * n_edges)
    Q2 = <float*> malloc(sizeof(float) * n_edges)
    attr_vecs = <float*> malloc(sizeof(float) * n_edges * dim)
    rep_vecs = <float*> malloc(sizeof(float) * n_edges * dim)
    local_attr_forces = <float*> malloc(sizeof(float) * n_vertices * dim)
    local_rep_forces = <float*> malloc(sizeof(float) * n_vertices * dim)
    with nogil:
        for v in prange(n_vertices):
            for d in range(dim):
                index = v * dim + d
                attractive_forces[index] = 0
                repulsive_forces[index] = 0

    cdef float grad = 0.0
    cdef float grad_d = 0.0
    with nogil:
        get_kernels(
            Q1,
            Q2,
            attr_vecs,
            rep_vecs,
            head,
            tail,
            head_embedding,
            tail_embedding,
            normalized,
            n_vertices,
            n_edges,
            dim,
            a,
            b
        )

        calc_forces(
            local_attr_forces,
            local_rep_forces,
            head,
            tail,
            weights,
            Q1,
            Q2,
            attr_vecs,
            rep_vecs,
            normalized,
            sym_attraction,
            n_vertices,
            n_edges,
            dim,
        )
        free(Q1)
        free(Q2)
        free(attr_vecs)
        free(rep_vecs)

        # Need to collect thread-local forces into single gradient
        #   before performing gradient descent
        with parallel():
            for v in prange(n_vertices):
                for d in range(dim):
                    index = v * dim + d
                    attractive_forces[index] += local_attr_forces[index]
                    repulsive_forces[index] += local_rep_forces[index]

        with parallel():
            for v in prange(n_vertices):
                for d in range(dim):
                    index = v * dim + d
                    grad_d = repulsive_forces[index] - attractive_forces[index]
                    head_embedding[v, d] += grad_d * lr
        free(local_attr_forces)
        free(local_rep_forces)

    free(forces)
    free(attractive_forces)
    free(repulsive_forces)



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
        float* forces,
        float* gains,
    forces = <float*> malloc(sizeof(float) * n_vertices * dim)
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

    free(forces)
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

