import numpy as py_np
cimport numpy as np
cimport cython
from libcpp cimport bool
from libc.stdio cimport printf
from libc.math cimport sqrt
from libc.stdlib cimport rand
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange, parallel

from sklearn.neighbors._quad_tree cimport _QuadTree

np.import_array()

cdef extern from "fastpow.c" nogil:
    double fastpow "fastPow" (double, double)
cdef extern from "fastpow.c" nogil:
    float fmax "my_fmax" (float, float)
cdef extern from "fastpow.c" nogil:
    float fmin "my_fmin" (float, float)
cdef extern from "fastpow.c" nogil:
    float fastsqrt "fastsqrt" (float)

ctypedef np.float32_t DTYPE_FLOAT
ctypedef np.int32_t DTYPE_INT

cdef float clip(float val, float lower, float upper) nogil:
    return fmax(lower, fmin(val, upper))

cdef float sq_euc_dist(float* x, float* y, int dim):
    """ squared euclidean distance between x and y """
    cdef float result = 0.0
    cdef float diff = 0
    cdef int i
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result

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

cdef float get_lr(initial_lr, i_epoch, n_epochs): 
    return initial_lr * (1.0 - (float(i_epoch) / float(n_epochs)))

cdef void print_status(i_epoch, n_epochs):
    if i_epoch % int(n_epochs / 10) == 0:
        print("Completed ", i_epoch, " / ", n_epochs, "epochs")

@cython.cdivision(True)
cdef float kernel_function(float dist_squared, float a, float b):
    if b <= 1:
        return 1 / (1 + a * fastpow(dist_squared, b))
    return fastpow(dist_squared, b - 1) / (1 + a * fastpow(dist_squared, b))

cdef float pos_force(
    int normalized,
    float p,
    float q,
    float Z
):
    if normalized:
        # FIXME - is it faster to get q^2 and then use that for q^3?
        return Z * p * (q ** 2 + 2 * q ** 3)
    return p * q ** 2

cdef float neg_force(
    int normalized,
    float q,
    float Z
):
    if normalized:
        return Z * (q ** 3 + 2 * q ** 4)
    return q ** 3

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _frob_umap_sampling(
    int normalized,
    int sym_attraction,
    np.ndarray[DTYPE_FLOAT, ndim=2] head_embedding,
    np.ndarray[DTYPE_FLOAT, ndim=2] tail_embedding,
    np.ndarray[DTYPE_INT, ndim=1] head,
    np.ndarray[DTYPE_INT, ndim=1] tail,
    np.ndarray[DTYPE_FLOAT, ndim=1] weights,
    np.ndarray[DTYPE_FLOAT, ndim=2] forces,
    np.ndarray[DTYPE_FLOAT, ndim=1] epochs_per_sample,
    float a,
    float b,
    int dim,
    int n_vertices,
    float lr,
    np.ndarray[DTYPE_FLOAT, ndim=1] epochs_per_negative_sample,
    np.ndarray[DTYPE_FLOAT, ndim=1] epoch_of_next_negative_sample,
    np.ndarray[DTYPE_FLOAT, ndim=1] epoch_of_next_sample,
    int i_epoch,
):
    cdef:
        int i, j, k, d, n_neg_samples
        float attractive_force, repulsive_force
        float dist_squared

    if normalized:
        raise ValueError("Cannot perform frobenius umap sampling with normalization")
    y1 = <float*> malloc(sizeof(float) * dim)
    y2 = <float*> malloc(sizeof(float) * dim)
    cdef float grad_d = 0.0
    cdef (float, float) rep_outputs
    cdef int n_edges = int(epochs_per_sample.shape[0])

    for i in range(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= i_epoch:
            # Gets one of the knn in HIGH-DIMENSIONAL SPACE relative to the sample point
            j = head[i]
            k = tail[i]

            for d in range(dim):
                y1[d] = head_embedding[j, d]
                y2[d] = tail_embedding[k, d]
            dist_squared = sq_euc_dist(y1, y2, dim)
            attractive_force = pos_force(
                normalized,
                weights[i],
                kernel_function(dist_squared, a, b),
                Z=1
            )

            for d in range(dim):
                grad_d = attractive_force * (y1[d] - y2[d])
                # Attractive force has a negative scalar on it
                #   in frobenius norm gradient
                head_embedding[j, d] -= grad_d * lr
                if sym_attraction:
                    head_embedding[k, d] += grad_d * lr

            epoch_of_next_sample[i] += epochs_per_sample[i]

            n_neg_samples = int(
                (i_epoch - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )
            for p in range(n_neg_samples):
                k = rand() % n_vertices
                for d in range(dim):
                    y2[d] = tail_embedding[k, d]
                dist_squared = sq_euc_dist(y1, y2, dim)
                repulsive_force = neg_force(
                    normalized,
                    kernel_function(dist_squared, a, b),
                    Z=1
                )

                for d in range(dim):
                    grad_d = repulsive_force * (y1[d] - y2[d])
                    head_embedding[j, d] += grad_d * lr

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )

def frob_umap_sampling(
    int normalized,
    int sym_attraction,
    int momentum,
    np.ndarray[DTYPE_FLOAT, ndim=2] head_embedding,
    np.ndarray[DTYPE_FLOAT, ndim=2] tail_embedding,
    np.ndarray[DTYPE_INT, ndim=1] head,
    np.ndarray[DTYPE_INT, ndim=1] tail,
    np.ndarray[DTYPE_FLOAT, ndim=1] weights,
    np.ndarray[DTYPE_FLOAT, ndim=2] forces,
    np.ndarray[DTYPE_FLOAT, ndim=2] gains,
    np.ndarray[DTYPE_FLOAT, ndim=1] epochs_per_sample,
    float a,
    float b,
    int dim,
    float initial_lr,
    float negative_sample_rate,
    int n_epochs,
    int n_vertices,
    bool verbose
):
    cdef:
        np.ndarray[DTYPE_FLOAT, ndim=1] epochs_per_negative_sample,
        np.ndarray[DTYPE_FLOAT, ndim=1] epoch_of_next_negative_sample,
        np.ndarray[DTYPE_FLOAT, ndim=1] epoch_of_next_sample,

    # ANDREW - perform negative samples x times more often
    #          by making the number of epochs between samples smaller
    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    # Make copies of these two
    epoch_of_next_negative_sample = py_np.ones_like(epochs_per_negative_sample) * epochs_per_negative_sample
    epoch_of_next_sample = py_np.ones_like(epochs_per_sample) * epochs_per_sample

    for i_epoch in range(n_epochs):
        lr = get_lr(initial_lr, i_epoch, n_epochs)
        _frob_umap_sampling(
            normalized,
            sym_attraction,
            head_embedding,
            tail_embedding,
            head,
            tail,
            weights,
            forces,
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _frob_umap_uniformly(
    int normalized,
    int sym_attraction,
    int momentum,
    np.ndarray[DTYPE_FLOAT, ndim=2] head_embedding,
    np.ndarray[DTYPE_FLOAT, ndim=2] tail_embedding,
    np.ndarray[DTYPE_INT, ndim=1] head,
    np.ndarray[DTYPE_INT, ndim=1] tail,
    np.ndarray[DTYPE_FLOAT, ndim=1] weights,
    np.ndarray[DTYPE_FLOAT, ndim=2] forces,
    np.ndarray[DTYPE_FLOAT, ndim=2] gains,
    float a,
    float b,
    int dim,
    int n_vertices,
    float lr,
    int i_epoch
):
    cdef:
        int i, j, k, d, v
        np.ndarray[DTYPE_FLOAT, ndim=2] attractive_forces
        np.ndarray[DTYPE_FLOAT, ndim=2] repulsive_forces
        np.ndarray[DTYPE_FLOAT, ndim=1] Q1
        np.ndarray[DTYPE_FLOAT, ndim=1] Q2
        np.ndarray[DTYPE_FLOAT, ndim=2] attr_vecs
        np.ndarray[DTYPE_FLOAT, ndim=2] rep_vecs
        float dist_squared, Z

    if normalized:
        Z = 0
    else:
        Z = 1

    attractive_forces = py_np.zeros([n_vertices, dim], dtype=py_np.float32)
    repulsive_forces = py_np.zeros([n_vertices, dim], dtype=py_np.float32)
    y1 = <float*> malloc(sizeof(float) * dim)
    y2 = <float*> malloc(sizeof(float) * dim)

    cdef float grad = 0.0
    cdef float grad_d = 0.0
    cdef int n_edges = int(head.shape[0])
    Q1 = py_np.zeros([n_edges], dtype=py_np.float32)
    Q2 = py_np.zeros([n_edges], dtype=py_np.float32)
    attr_vecs = py_np.zeros([n_edges, dim], dtype=py_np.float32)
    rep_vecs = py_np.zeros([n_edges, dim], dtype=py_np.float32)

    for edge in range(n_edges):
        j = head[edge]
        k = tail[edge]
        for d in range(dim):
            y1[d] = head_embedding[j, d]
            y2[d] = tail_embedding[k, d]
            attr_vecs[edge, d] = y1[d] - y2[d]
        dist_squared = sq_euc_dist(y1, y2, dim)
        Q1[edge] = kernel_function(dist_squared, a, b)

        k = rand() % n_vertices
        for d in range(dim):
            y2[d] = tail_embedding[k, d]
            rep_vecs[edge, d] = y1[d] - y2[d]
        dist_squared = sq_euc_dist(y1, y2, dim)
        Q2[edge] = kernel_function(dist_squared, a, b)
        if normalized:
            Z += Q2[edge]

    for edge in range(n_edges):
        j = head[edge]
        k = tail[edge]
        if normalized:
            Q1[edge] /= Z
            Q2[edge] /= Z
        grad = pos_force(normalized, weights[edge], Q1[edge], Z)
        for d in range(dim):
            grad_d = grad * attr_vecs[edge, d]
            attractive_forces[j, d] += grad_d
            if sym_attraction:
                attractive_forces[k, d] -= grad_d

        grad = neg_force(normalized, Q2[edge], Z)
        for d in range(dim):
            repulsive_forces[j, d] += grad * rep_vecs[edge, d]

    for v in range(n_vertices):
        for d in range(dim):
            grad_d = repulsive_forces[v, d] - attractive_forces[v, d]
            if momentum == 1:
                forces[v, d] = grad_d * lr + 0.9 * forces[v, d]
            else:
                forces[v, d] = grad_d * lr

            head_embedding[v, d] += forces[v, d]



def frob_umap_uniformly(
    int normalized,
    int sym_attraction,
    int momentum,
    np.ndarray[DTYPE_FLOAT, ndim=2] head_embedding,
    np.ndarray[DTYPE_FLOAT, ndim=2] tail_embedding,
    np.ndarray[DTYPE_INT, ndim=1] head,
    np.ndarray[DTYPE_INT, ndim=1] tail,
    np.ndarray[DTYPE_FLOAT, ndim=1] weights,
    np.ndarray[DTYPE_FLOAT, ndim=2] forces,
    np.ndarray[DTYPE_FLOAT, ndim=2] gains,
    np.ndarray[DTYPE_FLOAT, ndim=1] epochs_per_sample,
    float a,
    float b,
    int dim,
    float initial_lr,
    float negative_sample_rate,
    int n_epochs,
    int n_vertices,
    bool verbose
):
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
            forces,
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef _frob_pca(
    int normalized,
    int sym_attraction,
    int momentum,
    np.ndarray[DTYPE_FLOAT, ndim=2] head_embedding,
    np.ndarray[DTYPE_FLOAT, ndim=2] tail_embedding,
    np.ndarray[DTYPE_INT, ndim=1] head,
    np.ndarray[DTYPE_INT, ndim=1] tail,
    np.ndarray[DTYPE_FLOAT, ndim=1] weights,
    np.ndarray[DTYPE_FLOAT, ndim=2] forces,
    int dim,
    float lr,
    int n_vertices,
    int i_epoch
):
    cdef:
        int i, j, k, d, v
        np.ndarray[DTYPE_FLOAT, ndim=2] step_forces
        np.ndarray[DTYPE_FLOAT, ndim=1] y_dists
        np.ndarray[DTYPE_FLOAT, ndim=2] vecs
        float dist_squared, Z

    step_forces = py_np.zeros([n_vertices, dim], dtype=py_np.float32)
    y1 = <float*> malloc(sizeof(float) * dim)
    y2 = <float*> malloc(sizeof(float) * dim)

    cdef float mean_y_dist = 0.0
    cdef float grad = 0.0
    cdef float grad_d = 0.0
    cdef int n_edges = int(head.shape[0])
    y_dists = py_np.zeros([n_edges], dtype=py_np.float32)
    vecs = py_np.zeros([n_edges, dim], dtype=py_np.float32)

    for edge in range(n_edges):
        j = head[edge]
        k = tail[edge]
        for d in range(dim):
            y1[d] = head_embedding[j, d]
            y2[d] = tail_embedding[k, d]
            vecs[edge, d] = y1[d] - y2[d]
        y_dists[edge] = sq_euc_dist(y1, y2, dim)

    for edge in range(n_edges):
        mean_y_dist += y_dists[edge]
    mean_y_dist /= n_edges
    for edge in range(n_edges):
        y_dists[edge] = y_dists[edge] - mean_y_dist

    for edge in range(n_edges):
        j = head[edge]
        k = tail[edge]
        grad = weights[edge] - y_dists[edge]
        for d in range(dim):
            # grad_d = clip(grad * vecs[edge, d], -1, 1)
            grad_d = grad * vecs[edge, d]
            step_forces[j, d] += grad_d
            if sym_attraction:
                step_forces[k, d] -= grad_d

    for v in range(n_vertices):
        for d in range(dim):
            if momentum == 1:
                forces[v, d] = step_forces[v, d] * lr + 0.9 * forces[v, d]
            else:
                forces[v, d] = step_forces[v, d] * lr

            head_embedding[v, d] += forces[v, d]



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def frob_pca(
    int normalized,
    int sym_attraction,
    int momentum,
    np.ndarray[DTYPE_FLOAT, ndim=2] head_embedding,
    np.ndarray[DTYPE_FLOAT, ndim=2] tail_embedding,
    np.ndarray[DTYPE_INT, ndim=1] head,
    np.ndarray[DTYPE_INT, ndim=1] tail,
    np.ndarray[DTYPE_FLOAT, ndim=1] weights,
    np.ndarray[DTYPE_FLOAT, ndim=2] forces,
    np.ndarray[DTYPE_FLOAT, ndim=2] gains,
    np.ndarray[DTYPE_FLOAT, ndim=1] epochs_per_sample,
    float a,
    float b,
    int dim,
    float initial_lr,
    float negative_sample_rate,
    int n_epochs,
    int n_vertices,
    bool verbose
):
    cdef float mean_weight = 0.0
    for weight in weights:
        mean_weight += weight
    mean_weight /= int(weights.shape[0])
    for i, weight in enumerate(weights):
        weights[i] = weight - mean_weight

    for i_epoch in range(n_epochs):
        lr = get_lr(initial_lr, i_epoch, n_epochs)
        _frob_pca(
            normalized,
            sym_attraction,
            momentum,
            head_embedding,
            tail_embedding,
            head,
            tail,
            weights,
            forces,
            dim,
            lr,
            n_vertices,
            i_epoch
        )
        if verbose:
            print_status(i_epoch, n_epochs)


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
    bool verbose=True,
    **kwargs
):
    cdef:
        int dim, i_epoch
        int n_edges
        np.ndarray[DTYPE_FLOAT, ndim=2] forces
        np.ndarray[DTYPE_FLOAT, ndim=2] gains

    dim = head_embedding.shape[1]
    forces = py_np.zeros([n_vertices, dim], dtype=py_np.float32)
    gains = py_np.ones([n_vertices, dim], dtype=py_np.float32)

    # Perform weight scaling on high-dimensional relationships
    cdef float weight_sum = 0.0
    if normalized:
        for i in range(weights.shape[0]):
            weight_sum = weight_sum + weights[i]
        for i in range(weights.shape[0]):
            weights[i] = weights[i] / weight_sum
        initial_lr *= 2000000
    # FIXME
    if 'pca' in optimize_method:
        initial_lr = 0.0005

    optimize_dict = {
        'frob_umap_uniform': frob_umap_uniformly,
        'frob_pca': frob_pca,
        'frob_umap_sampling': frob_umap_sampling,
    }
    optimize_fn = optimize_dict[optimize_method]

    optimize_fn(
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

