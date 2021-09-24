import numpy as np
cimport numpy as np
from libcpp cimport bool
from libc.stdio cimport printf
from libc.math cimport sqrt, log
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange, parallel

from sklearn.neighbors._quad_tree cimport _QuadTree

np.import_array()

cdef clip(float val):
    """Standard clamping of a value into a fixed range (in this case -4.0 to
    4.0)

    Parameters
    ----------
    val: float
        The value to be clamped.

    Returns
    -------
    The clamped value, now fixed to be in the range -4.0 to 4.0.
    """
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val


cdef rdist(float* x, float* y, int dim):
    """Reduced Euclidean distance.

    Parameters
    ----------
    x: array of shape (embedding_dim,)
    y: array of shape (embedding_dim,)

    Returns
    -------
    The squared euclidean distance between x and y
    """
    cdef float result = 0.0
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result


cdef umap_pos_force_kernel(float dist_squared, float a, float b):
    cdef float grad_scalar
    grad_scalar = -2.0 * a * b * pow(dist_squared, b - 1.0)
    grad_scalar /= a * pow(dist_squared, b) + 1.0
    return grad_scalar

cdef umap_neg_force_kernel(float dist_squared, float a, float b):
    # This is the repulsive force for umap
    cdef float phi_ijZ
    phi_ijZ = 2.0 * b
    phi_ijZ /= (0.001 + dist_squared) * (a * pow(dist_squared, b) + 1)
    return phi_ijZ

cdef tsne_kernel(float dist_squared):
    return 1 / (1 + dist_squared)

cdef calculate_barnes_hut(
        str weight_scaling,
        str kernel_choice,
        np.float32_t[:, :] head_embedding,
        np.float32_t[:, :] tail_embedding,
        np.int32_t[:] head,
        np.int32_t[:] tail,
        np.float32_t[:] weights,
        np.float64_t[:, :] grads,
        np.float64_t[:] epochs_per_sample,
        _QuadTree qt,
        float a,
        float b,
        int dim,
        int n_vertices,
        float alpha,
):
    cdef:
        double qijZ, sum_Q = 0
        int i, j, k, l
        float cell_size, cell_dist, grad_scalar
        long offset = dim + 2

    # Allocte memory for data structures
    cell_summaries = <float*> malloc(sizeof(float) * n_vertices * offset)
    y1 = <float*> malloc(sizeof(float) * dim)
    y2 = <float*> malloc(sizeof(float) * dim)
    # FIXME - what type should these be for optimal performance?
    cdef np.ndarray[double, mode="c", ndim=2] pos_grads = np.zeros([n_vertices, dim])
    cdef np.ndarray[double, mode="c", ndim=2] neg_grads = np.zeros([n_vertices, dim])
    cdef np.ndarray[double, mode="c", ndim=2] all_grads = np.zeros([n_vertices, dim])

    cdef int num_edges = int(epochs_per_sample.shape[0])
    cdef float average_weight = np.sum(weights) / num_edges

    # FIXME - rename epochs_per_sample
    for i in range(epochs_per_sample.shape[0]):
        # Get one of the knn edges
        j = head[i]
        k = tail[i]

        for d in range(dim):
            # FIXME - index should be typed for more efficient access
            y1[d] = head_embedding[j, d]
            y2[d] = tail_embedding[k, d]
        dist_squared = rdist(y1, y2, dim)

        if kernel_choice == 'umap':
            grad_scalar = umap_pos_force_kernel(dist_squared, a, b)
        else:
            assert kernel_choice == 'tsne'
            grad_scalar = tsne_kernel(dist_squared)
        grad_scalar *= weights[i]

        for d in range(dim):
            pos_grad = clip(grad_scalar * (y1[d] - y2[d]))
            pos_grads[j][d] += pos_grad

    for v in range(n_vertices):
        # Get necessary data regarding current point and the quadtree cells
        for d in range(dim):
            y1[d] = head_embedding[v, d]
        cell_metadata = qt.summarize(y1, cell_summaries, 0.25) # 0.25 = theta^2

        # For each quadtree cell with respect to the current point
        for i_cell in range(cell_metadata // offset):
            cell_dist = cell_summaries[i_cell * offset + dim]
            cell_size = cell_summaries[i_cell * offset + dim + 1]

            if kernel_choice == 'umap':
                kernel = umap_neg_force_kernel(cell_dist, a, b)
            else:
                assert kernel_choice == 'tsne'
                kernel = tsne_kernel(cell_dist)

            if weight_scaling == 'umap':
                kernel *= (1 - average_weight)
                grad_scalar = cell_size * kernel
            else:
                assert weight_scaling == 'tsne'
                sum_Q += cell_size * kernel # Collect the q_ij's contributions into Z
                grad_scalar = cell_size * kernel * kernel

            for d in range(dim):
                neg_grads[v][d] += grad_scalar * cell_summaries[i_cell * offset + d]

    if weight_scaling == 'tsne':
        # normalize low-dim weight matrix by matrix sum
        neg_grads *= 4 / sum_Q
        pos_grads *= 4

    for v in range(n_vertices):
        for d in range(dim):
            grads[v][d] = pos_grads[v][d] - neg_grads[v][d]
            head_embedding[v][d] -= grads[v][d] * alpha

    return grads

def bh_wrapper(
        str weight_scaling,
        str kernel_choice,
        np.float32_t[:, :] head_embedding,
        np.float32_t[:, :] tail_embedding,
        np.int32_t[:] head,
        np.int32_t[:] tail,
        np.float32_t[:] weights,
        np.float64_t[:, :] grads,
        np.float64_t[:] epochs_per_sample,
        float a,
        float b,
        int dim,
        int n_vertices,
        float alpha,
):
    # Can only define cython quadtree in a cython function
    cdef _QuadTree qt = _QuadTree(dim, 1)
    qt.build_tree(head_embedding)
    return calculate_barnes_hut(
        weight_scaling,
        kernel_choice,
        head_embedding,
        tail_embedding,
        head,
        tail,
        weights,
        grads,
        epochs_per_sample,
        qt,
        a,
        b,
        dim,
        n_vertices,
        alpha
    )
