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

###################
##### KERNELS #####
###################

cdef umap_pos_force_kernel(float dist_squared, float a, float b):
    cdef float grad_scalar = 0.0
    if dist_squared > 0.0:
        grad_scalar = -2.0 * a * b * pow(dist_squared, b - 1.0)
        grad_scalar /= a * pow(dist_squared, b) + 1.0
    return grad_scalar

cdef umap_neg_force_kernel(float dist_squared, float a, float b):
    cdef float phi_ijZ = 0.0
    if dist_squared > 0.0:
        phi_ijZ = 2.0 * b
        phi_ijZ /= (0.001 + dist_squared) * (a * pow(dist_squared, b) + 1)
    return phi_ijZ

cdef tsne_kernel(float dist_squared):
    return 1 / (1 + dist_squared)

def pos_force_kernel(str kernel_choice, float dist_squared, float a, float b):
    if kernel_choice == 'umap':
        return umap_pos_force_kernel(dist_squared, a, b)
    assert kernel_choice == 'tsne'
    return tsne_kernel(dist_squared)

def neg_force_kernel(str kernel_choice, float dist_squared, float a, float b):
    if kernel_choice == 'umap':
        return umap_neg_force_kernel(dist_squared, a, b)
    assert kernel_choice == 'tsne'
    return tsne_kernel(dist_squared)

###################
##### WEIGHTS #####
###################

cdef umap_pos_weight_scaling(
        np.float32_t[:] weights,
        initial_alpha
    ):
    return weights, initial_alpha

cdef tsne_pos_weight_scaling(
        np.float32_t[:] weights,
        initial_alpha
    ):
    # FIXME - early exaggeration!!
    return weights / np.sum(weights), initial_alpha * 200

def pos_weight_scaling(
        str weight_scaling_choice,
        np.float32_t[:] weights,
        initial_alpha
    ):
    if weight_scaling_choice == 'umap':
        # umap doesn't scale P_ij weights
        return umap_pos_weight_scaling(weights, initial_alpha)
    assert weight_scaling_choice == 'tsne'
    return tsne_pos_weight_scaling(weights, initial_alpha)


cdef umap_neg_weight_scaling(
        float kernel,
        int cell_size,
        float average_weight,
    ):
    neg_force = kernel * (1 - average_weight) * cell_size
    return neg_force, 0.0

cdef tsne_neg_weight_scaling(
        float kernel,
        int cell_size,
        float weight_scalar
    ):
    weight_scalar += cell_size * kernel # Collect the q_ij's contributions into Z
    neg_force = cell_size * kernel * kernel
    return neg_force, weight_scalar

def neg_weight_scaling(
        str weight_scaling_choice,
        float kernel,
        int cell_size,
        float average_weight,
        float weight_scalar
    ):
    if weight_scaling_choice == 'umap':
        return umap_neg_weight_scaling(kernel, cell_size, average_weight)
    assert weight_scaling_choice == 'tsne'
    return tsne_neg_weight_scaling(kernel, cell_size, weight_scalar)


cdef umap_total_weight_scaling(pos_grads, neg_grads):
    return pos_grads, neg_grads

cdef tsne_total_weight_scaling(pos_grads, neg_grads, float weight_scalar):
    neg_grads *= 4 / weight_scalar
    pos_grads *= 4
    return pos_grads, neg_grads

def total_weight_scaling(
        str weight_scaling_choice,
        pos_grads,
        neg_grads,
        float weight_scalar,
    ):
    if weight_scaling_choice == 'umap':
        return umap_total_weight_scaling(pos_grads, neg_grads)
    assert weight_scaling_choice == 'tsne'
    return tsne_total_weight_scaling(pos_grads, neg_grads, weight_scalar)


##### BARNES-HUT CODE #####

cdef calculate_barnes_hut(
        str weight_scaling_choice,
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
        double weight_scalar = 0.0
        int i, j, k, l, num_cells
        float cell_size, cell_dist, grad_scalar
        long offset = dim + 2
        long dim_index

    # Allocte memory for data structures
    cell_summaries = <float*> malloc(sizeof(float) * n_vertices * offset)
    y1 = <float*> malloc(sizeof(float) * dim)
    y2 = <float*> malloc(sizeof(float) * dim)
    # FIXME - what type should these be for optimal performance?
    cdef np.ndarray[double, mode="c", ndim=2] pos_grads = np.zeros([n_vertices, dim])
    cdef np.ndarray[double, mode="c", ndim=2] neg_grads = np.zeros([n_vertices, dim])
    cdef np.ndarray[double, mode="c", ndim=2] all_grads = np.zeros([n_vertices, dim])

    cdef int n_edges = int(epochs_per_sample.shape[0])
    cdef float average_weight = np.sum(weights) / n_edges

    for edge in range(n_edges):
        # Get vertices on either side of the edge
        j = head[edge] # head is the incoming data being transformed
        k = tail[edge] # tail is what we fit to

        for d in range(dim):
            # FIXME - index should be typed for more efficient access
            y1[d] = head_embedding[j, d]
            y2[d] = tail_embedding[k, d]
        dist_squared = rdist(y1, y2, dim)
        pos_kernel = pos_force_kernel(weight_scaling_choice, dist_squared, a, b)
        pos_force = pos_kernel * weights[edge]
        for d in range(dim):
            pos_grads[j, d] += clip(pos_force * (y1[d] - y2[d]))

    for v in range(n_vertices):
        # Get necessary data regarding current point and the quadtree cells
        for d in range(dim):
            y1[d] = head_embedding[v, d]
        cell_metadata = qt.summarize(y1, cell_summaries, 0.25) # 0.25 = theta^2
        num_cells = cell_metadata // offset
        cell_sizes = [cell_summaries[i * offset + dim + 1] for i in range(num_cells)]

        # For each quadtree cell with respect to the current point
        for i_cell in range(num_cells):
            cell_dist = cell_summaries[i_cell * offset + dim]
            cell_size = cell_summaries[i_cell * offset + dim + 1]
            # FIXME - think more about this cell_size bounding
            # Gives clusters that REALLY preserve local relationships while 
            #   generally maintaining global ones
            if cell_size < 1:
                continue
            neg_kernel = neg_force_kernel(weight_scaling_choice, cell_dist, a, b)
            neg_force, weight_scalar = neg_weight_scaling(
                weight_scaling_choice,
                neg_kernel,
                cell_size,
                average_weight,
                weight_scalar
            )
            for d in range(dim):
                dim_index = i_cell * offset + d
                neg_grads[v][d] += neg_force * cell_summaries[dim_index]

    pos_grads, neg_grads = total_weight_scaling(
        weight_scaling_choice,
        pos_grads,
        neg_grads,
        weight_scalar,
    )

    for v in range(n_vertices):
        for d in range(dim):
            grads[v][d] = (pos_grads[v][d] - neg_grads[v][d]) + 0.9 * grads[v][d]
            head_embedding[v][d] -= grads[v][d] * alpha

    return grads

def bh_wrapper(
        str weight_scaling_choice,
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
    """
    Wrapper to call barnes_hut optimization
    Require a regular def function to call from python file
    But this standard function in a .pyx file can call the cdef function
    """
    # Can only define cython quadtree in a cython function
    cdef _QuadTree qt = _QuadTree(dim, 1)
    qt.build_tree(head_embedding)
    return calculate_barnes_hut(
        weight_scaling_choice,
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
