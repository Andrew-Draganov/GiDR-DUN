import numpy as py_np
cimport numpy as np
cimport cython
from libcpp cimport bool
from libc.stdio cimport printf
from libc.stdlib cimport rand
from libc.math cimport pow
from libc.math cimport sqrt, log
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange, parallel

from sklearn.neighbors._quad_tree cimport _QuadTree

np.import_array()

ctypedef np.float32_t DTYPE_FLOAT
ctypedef np.float64_t DTYPE_FLOAT64
ctypedef np.int32_t DTYPE_INT

STUFF = 'hiii'

cdef float clip(float val):
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


cdef float rdist(float* x, float* y, int dim):
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
    cdef float diff = 0
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result

###################
##### KERNELS #####
###################

@cython.cdivision(True)
cdef float umap_attraction_grad(float dist_squared, float a, float b):
    cdef float grad_scalar = 0.0
    grad_scalar = -2.0 * a * b * pow(dist_squared, b - 1.0)
    grad_scalar /= a * pow(dist_squared, b) + 1.0
    return grad_scalar

@cython.cdivision(True)
cdef float umap_repulsion_grad(float dist_squared, float a, float b):
    cdef float phi_ijZ = 0.0
    phi_ijZ = 2.0 * b
    phi_ijZ /= (0.001 + dist_squared) * (a * pow(dist_squared, b) + 1)
    return phi_ijZ

@cython.cdivision(True)
cdef float kernel_function(float dist_squared, float a, float b):
    return 1 / (1 + a * pow(dist_squared, b))

###################
##### WEIGHTS #####
###################

cdef umap_p_scaling(
        np.float32_t[:] weights,
        float initial_alpha
    ):
    return weights, initial_alpha

cdef tsne_p_scaling(
        np.ndarray[DTYPE_FLOAT, ndim=1] weights,
        float initial_alpha
    ):
    # FIXME - add early exaggeration!!
    cdef float weight_sum = 0.0
    for i in range(weights.shape[0]):
        weight_sum = weight_sum + weights[i]
    for i in range(weights.shape[0]):
        weights[i] = weights[i] / weight_sum
    return weights, initial_alpha * 200

cdef p_scaling(
        str normalization,
        np.ndarray[DTYPE_FLOAT, ndim=1] weights,
        float initial_alpha
    ):
    if normalization == 'umap':
        # umap doesn't scale P_ij weights
        return umap_p_scaling(weights, initial_alpha)
    assert normalization == 'tsne'
    return tsne_p_scaling(weights, initial_alpha)


cdef (float, float) umap_repulsive_force(
        float dist_squared,
        float a,
        float b,
        int cell_size,
        float average_weight,
    ):
    cdef:
        float repulsive_force
    # ANDREW - Using average_weight is a lame approximation
    #        - Realistically, we should use the actual weight on
    #          the edge e_{ik}, but the coo_matrix is not
    #          indexable. So we assume the differences cancel out over
    #          enough iterations
    cdef float kernel = umap_repulsion_grad(dist_squared, a, b)
    repulsive_force = cell_size * kernel * (1 - average_weight)
    return (repulsive_force, 0)

cdef (float, float) tsne_repulsive_force(
        float dist_squared,
        float a,
        float b,
        int cell_size,
        float Z
    ):
    cdef float kernel = kernel_function(dist_squared, a, b)
    Z += cell_size * kernel # Collect the q_ij's contributions into Z
    cdef float repulsive_force = cell_size * kernel * kernel
    return (repulsive_force, Z)

cdef float attractive_force_func(
        str normalization,
        float dist_squared,
        float a,
        float b,
        float edge_weight
    ):
    if normalization == 'umap':
        edge_force = umap_attraction_grad(dist_squared, a, b)
    else:
        assert normalization == 'tsne'
        edge_force = kernel_function(dist_squared, a, b)

    # FIXME FIXME FIXME
    # This does NOT work with parallel=True
    return edge_force * edge_weight

cdef (float, float) repulsive_force_func(
        str normalization,
        float dist_squared,
        float a,
        float b,
        int cell_size,
        float average_weight,
        float Z
    ):
    if normalization == 'umap':
        return umap_repulsive_force(
            dist_squared,
            a,
            b,
            cell_size,
            average_weight
        )
    assert normalization == 'tsne'
    return tsne_repulsive_force(
        dist_squared,
        a,
        b,
        cell_size,
        Z
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _cy_umap_sampling(
    str normalization,
    np.ndarray[DTYPE_FLOAT, ndim=2] head_embedding,
    np.ndarray[DTYPE_FLOAT, ndim=2] tail_embedding,
    np.ndarray[DTYPE_INT, ndim=1] head,
    np.ndarray[DTYPE_INT, ndim=1] tail,
    np.ndarray[DTYPE_FLOAT, ndim=1] weights,
    np.ndarray[DTYPE_FLOAT64, ndim=2] grads,
    np.ndarray[DTYPE_FLOAT64, ndim=1] epochs_per_sample,
    float a,
    float b,
    int dim,
    int n_vertices,
    float alpha,
    np.ndarray[DTYPE_FLOAT64, ndim=1] epochs_per_negative_sample,
    np.ndarray[DTYPE_FLOAT64, ndim=1] epoch_of_next_negative_sample,
    np.ndarray[DTYPE_FLOAT64, ndim=1] epoch_of_next_sample,
    int i_epoch
):
    cdef:
        int i, j, k, d, n_neg_samples
        float attractive_force, repulsive_force
        float dist_squared
        float Z = 0.0

    y1 = <float*> malloc(sizeof(float) * dim)
    y2 = <float*> malloc(sizeof(float) * dim)
    cdef float grad_d = 0.0
    cdef (float, float) rep_outputs
    cdef int n_edges = int(epochs_per_sample.shape[0])
    cdef float average_weight = 0.0
    for i in range(weights.shape[0]):
        average_weight = average_weight + weights[i]
    average_weight = average_weight / n_edges

    for i in range(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= i_epoch:
            # Gets one of the knn in HIGH-DIMENSIONAL SPACE relative to the sample point
            j = head[i]
            k = tail[i]

            if j == k:
                continue
            for d in range(dim):
                # FIXME - index should be typed for more efficient access
                y1[d] = head_embedding[j, d]
                y2[d] = tail_embedding[k, d]
            # ANDREW - optimize positive force for each edge
            dist_squared = rdist(y1, y2, dim)
            attractive_force = attractive_force_func(
                normalization,
                dist_squared,
                a,
                b,
                weights[i]
            )

            for d in range(dim):
                grad_d = clip(attractive_force * (y1[d] - y2[d]))
                head_embedding[j, d] += grad_d * alpha
                head_embedding[k, d] -= grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]

            # ANDREW - Picks random vertex from ENTIRE graph and calculates repulsive force
            # ANDREW - If we are summing the effects of the forces and multiplying them
            #   by the weights appropriately, we only need to alternate symmetrically
            #   between positive and negative forces rather than doing 1 positive
            #   calculation to n negative ones
            # FIXME - add random seed option
            n_neg_samples = int(
                (i_epoch - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )
            for p in range(n_neg_samples):
                k = rand() % n_vertices
                if j == k:
                    continue
                for d in range(dim):
                    y2[d] = tail_embedding[k, d]
                dist_squared = rdist(y1, y2, dim)
                rep_outputs = repulsive_force_func(
                    normalization,
                    dist_squared,
                    a,
                    b,
                    cell_size=1,
                    average_weight=average_weight,
                    Z=Z
                )
                repulsive_force = rep_outputs[0]
                Z = rep_outputs[1]

                for d in range(dim):
                    if repulsive_force > 0.0:
                        grad_d = clip(repulsive_force * (y1[d] - y2[d]))
                    else:
                        grad_d = 4.0
                    head_embedding[j, d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )

def cy_umap_sampling(
    str normalization,
    np.ndarray[DTYPE_FLOAT, ndim=2] head_embedding,
    np.ndarray[DTYPE_FLOAT, ndim=2] tail_embedding,
    np.ndarray[DTYPE_INT, ndim=1] head,
    np.ndarray[DTYPE_INT, ndim=1] tail,
    np.ndarray[DTYPE_FLOAT, ndim=1] weights,
    np.ndarray[DTYPE_FLOAT64, ndim=2] grads,
    np.ndarray[DTYPE_FLOAT64, ndim=1] epochs_per_sample,
    float a,
    float b,
    int dim,
    int n_vertices,
    float alpha,
    np.ndarray[DTYPE_FLOAT64, ndim=1] epochs_per_negative_sample,
    np.ndarray[DTYPE_FLOAT64, ndim=1] epoch_of_next_negative_sample,
    np.ndarray[DTYPE_FLOAT64, ndim=1] epoch_of_next_sample,
    int i_epoch
):
    _cy_umap_sampling(
        normalization,
        head_embedding,
        tail_embedding,
        head,
        tail,
        weights,
        grads,
        epochs_per_sample,
        a,
        b,
        dim,
        n_vertices,
        alpha,
        epochs_per_negative_sample,
        epoch_of_next_negative_sample,
        epoch_of_next_sample,
        i_epoch
    )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _cy_umap_uniformly(
    str normalization,
    np.ndarray[DTYPE_FLOAT, ndim=2] head_embedding,
    np.ndarray[DTYPE_FLOAT, ndim=2] tail_embedding,
    np.ndarray[DTYPE_INT, ndim=1] head,
    np.ndarray[DTYPE_INT, ndim=1] tail,
    np.ndarray[DTYPE_FLOAT, ndim=1] weights,
    np.ndarray[DTYPE_FLOAT64, ndim=2] grads,
    np.ndarray[DTYPE_FLOAT64, ndim=1] epochs_per_sample,
    float a,
    float b,
    int dim,
    int n_vertices,
    float alpha,
):
    cdef:
        int i, j, k, d
        np.ndarray[DTYPE_FLOAT, ndim=2] attraction
        np.ndarray[DTYPE_FLOAT, ndim=2] repulsion
        float Z = 0.0
        float attractive_force, repulsive_force
        float dist_squared

    attraction = py_np.zeros([n_vertices, dim], dtype=py_np.float32)
    repulsion = py_np.zeros([n_vertices, dim], dtype=py_np.float32)
    y1 = <float*> malloc(sizeof(float) * dim)
    y2 = <float*> malloc(sizeof(float) * dim)
    cdef float grad_d = 0.0
    cdef (float, float) rep_outputs
    cdef int n_edges = int(epochs_per_sample.shape[0])
    cdef float average_weight = 0.0
    for i in range(weights.shape[0]):
        average_weight += weights[i]
    average_weight /= n_edges

    for i in range(epochs_per_sample.shape[0]):
        # Gets one of the knn in HIGH-DIMENSIONAL SPACE relative to the sample point
        j = head[i]
        k = tail[i]
        if j == k:
            continue

        for d in range(dim):
            # FIXME - index should be typed for more efficient access
            y1[d] = head_embedding[j, d]
            y2[d] = tail_embedding[k, d]
        # ANDREW - optimize positive force for each edge
        dist_squared = rdist(y1, y2, dim)
        attractive_force = attractive_force_func(
            normalization,
            dist_squared,
            a,
            b,
            weights[i]
        )

        for d in range(dim):
            grad_d = clip(attractive_force * (y1[d] - y2[d]))
            attraction[j, d] += grad_d
            attraction[k, d] -= grad_d

        # ANDREW - Picks random vertex from ENTIRE graph and calculates repulsive force
        # ANDREW - If we are summing the effects of the forces and multiplying them
        #   by the weights appropriately, we only need to alternate symmetrically
        #   between positive and negative forces rather than doing 1 positive
        #   calculation to n negative ones
        # FIXME - add random seed option
        k = rand() % n_vertices
        if j == k:
            continue
        for d in range(dim):
            y2[d] = tail_embedding[k, d]
        dist_squared = rdist(y1, y2, dim)
        rep_outputs = repulsive_force_func(
            normalization,
            dist_squared,
            a,
            b,
            cell_size=1,
            average_weight=average_weight,
            Z=Z
        )
        repulsive_force = rep_outputs[0]
        Z = rep_outputs[1]

        for d in range(dim):
            if repulsive_force > 0.0:
                grad_d = clip(repulsive_force * (y1[d] - y2[d]))
            else:
                grad_d = 4.0
            repulsion[j, d] += grad_d

    cdef float rep_scalar = -4
    if normalization == 'tsne':
        rep_scalar = rep_scalar / Z
    cdef float att_scalar = 4
    for v in range(n_vertices):
        for d in range(dim):
            if normalization == 'tsne':
                repulsion[v, d] = repulsion[v, d] * rep_scalar
                attraction[v, d] = attraction[v, d] * att_scalar

            head_embedding[v, d] += (attraction[v, d] + repulsion[v, d]) * alpha

def cy_umap_uniformly(
    str normalization,
    np.ndarray[DTYPE_FLOAT, ndim=2] head_embedding,
    np.ndarray[DTYPE_FLOAT, ndim=2] tail_embedding,
    np.ndarray[DTYPE_INT, ndim=1] head,
    np.ndarray[DTYPE_INT, ndim=1] tail,
    np.ndarray[DTYPE_FLOAT, ndim=1] weights,
    np.ndarray[DTYPE_FLOAT64, ndim=2] grads,
    np.ndarray[DTYPE_FLOAT64, ndim=1] epochs_per_sample,
    float a,
    float b,
    int dim,
    int n_vertices,
    float alpha,
):
    _cy_umap_uniformly(
        normalization,
        head_embedding,
        tail_embedding,
        head,
        tail,
        weights,
        grads,
        epochs_per_sample,
        a,
        b,
        dim,
        n_vertices,
        alpha,
    )

# 
# 
# 
# ##### BARNES-HUT CODE #####
# 
# cdef calculate_barnes_hut(
#         str normalization,
#         np.float32_t[:, :] head_embedding,
#         np.float32_t[:, :] tail_embedding,
#         np.int32_t[:] head,
#         np.int32_t[:] tail,
#         np.float32_t[:] weights,
#         np.float64_t[:, :] grads,
#         np.float64_t[:] epochs_per_sample,
#         _QuadTree qt,
#         float a,
#         float b,
#         int dim,
#         int n_vertices,
#         float alpha,
# ):
#     cdef:
#         double Z = 0.0
#         int i, j, k, l, num_cells, d
#         float cell_size, cell_dist, grad_scalar, dist_squared
#         long offset = dim + 2
#         long dim_index
# 
#     # Allocte memory for data structures
#     cell_summaries = <float*> malloc(sizeof(float) * n_vertices * offset)
#     y1 = <float*> malloc(sizeof(float) * dim)
#     y2 = <float*> malloc(sizeof(float) * dim)
#     # FIXME - what type should these be for optimal performance?
#     cdef attractive_forces = np.zeros([n_vertices, dim], dtype=np.float32)
#     cdef repulsive_forces = np.zeros([n_vertices, dim], dtype=np.float32)
#     cdef forces = np.zeros([n_vertices, dim], dtype=np.float32)
# 
#     cdef int n_edges = int(epochs_per_sample.shape[0])
#     cdef float average_weight = 0.0
#     for i in range(weights.shape[0]):
#         average_weight += weights[i]
#     average_weight /= n_edges
# 
#     for edge in range(n_edges):
#         # Get vertices on either side of the edge
#         j = head[edge] # head is the incoming data being transformed
#         k = tail[edge] # tail is what we fit to
# 
#         for d in range(dim):
#             # FIXME - index should be typed for more efficient access
#             y1[d] = head_embedding[j, d]
#             y2[d] = tail_embedding[k, d]
#         dist_squared = rdist(y1, y2, dim)
#         attractive_force = attractive_force_func(
#             normalization,
#             dist_squared,
#             a,
#             b,
#             weights[edge]
#         )
#         for d in range(dim):
#             attractive_forces[j, d] += clip(attractive_force * (y1[d] - y2[d]))
# 
#     for v in range(n_vertices):
#         # Get necessary data regarding current point and the quadtree cells
#         for d in range(dim):
#             y1[d] = head_embedding[v, d]
#         cell_metadata = qt.summarize(y1, cell_summaries, 0.25) # 0.25 = theta^2
#         num_cells = cell_metadata // offset
#         cell_sizes = [cell_summaries[i * offset + dim + 1] for i in range(num_cells)]
# 
#         # For each quadtree cell with respect to the current point
#         for i_cell in range(num_cells):
#             cell_dist = cell_summaries[i_cell * offset + dim]
#             cell_size = cell_summaries[i_cell * offset + dim + 1]
#             # FIXME - think more about this cell_size bounding
#             # Ignoring small cells gives clusters that REALLY preserve 
#             #      local relationships while generally maintaining global ones
#             if cell_size < 1:
#                 continue
#             repulsive_force, Z = repulsive_force_func(
#                 normalization,
#                 dist_squared,
#                 a,
#                 b,
#                 int(cell_size),
#                 average_weight,
#                 Z
#             )
#             for d in range(dim):
#                 dim_index = i_cell * offset + d
#                 repulsive_forces[v][d] += repulsive_force * cell_summaries[dim_index]
# 
#     if normalization == 'tsne':
#         cdef float rep_scalar = -4 / Z
#         cdef float att_scalar = 4
#         for i in range(repulsion.shape[0]):
#             for j in range(repulsion.shape[1]):
#                 repulsion[i, j] = repulsion[i, j] * rep_scalar
#                 attraction[i, j] = attraction[i, j] * att_scalar
# 
#     for v in range(n_vertices):
#         for d in range(dim):
#             forces[v][d] = (attractive_forces[v][d] + repulsive_forces[v][d]) + 0.9 * forces[v][d]
#             head_embedding[v][d] -= forces[v][d] * alpha
# 
#     return grads
# 
# def bh_wrapper(
#         str opt_method,
#         str normalization,
#         np.float32_t[:, :] head_embedding,
#         np.float32_t[:, :] tail_embedding,
#         np.int32_t[:] head,
#         np.int32_t[:] tail,
#         np.float32_t[:] weights,
#         np.float64_t[:, :] grads,
#         np.float64_t[:] epochs_per_sample,
#         float a,
#         float b,
#         int dim,
#         int n_vertices,
#         float alpha,
# ):
#     """
#     Wrapper to call barnes_hut optimization
#     Require a regular def function to call from python file
#     But this standard function in a .pyx file can call the cdef function
#     """
#     # Can only define cython quadtree in a cython function
#     cdef _QuadTree qt = _QuadTree(dim, 1)
#     qt.build_tree(head_embedding)
#     return calculate_barnes_hut(
#         normalization,
#         head_embedding,
#         tail_embedding,
#         head,
#         tail,
#         weights,
#         grads,
#         epochs_per_sample,
#         qt,
#         a,
#         b,
#         dim,
#         n_vertices,
#         alpha
#     )
