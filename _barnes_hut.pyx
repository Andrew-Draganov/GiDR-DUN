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


cdef calculate_barnes_hut_umap(
        np.float32_t[:, :] head_embedding,
        np.float32_t[:, :] tail_embedding,
        np.int32_t[:] head,
        np.int32_t[:] tail,
        np.float32_t[:] weights,
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
    summary = <float*> malloc(sizeof(float) * n_vertices * offset)
    current = <float*> malloc(sizeof(float) * dim)
    other = <float*> malloc(sizeof(float) * dim)
    cdef np.ndarray[double, mode="c", ndim=2] pos_grads = np.zeros([n_vertices, dim])
    cdef np.ndarray[double, mode="c", ndim=2] neg_grads = np.zeros([n_vertices, dim])
    cdef np.ndarray[double, mode="c", ndim=2] all_grads = np.zeros([n_vertices, dim])

    cdef int num_edges = int(epochs_per_sample.shape[0])
    cdef float average_weight = np.sum(weights) / num_edges

    for i in range(epochs_per_sample.shape[0]):
        # Gets one of the knn in HIGH-DIMENSIONAL SPACE relative to the sample point
        j = head[i]
        k = tail[i]

        for d in range(dim):
            # FIXME - index should be typed for more efficient access
            current[d] = head_embedding[j, d]
            other[d] = tail_embedding[k, d]
        dist_squared = rdist(current, other, dim)

        if dist_squared > 0.0:
            # ANDREW - this is the actual attractive force for UMAP
            grad_scalar = -2.0 * a * b * pow(dist_squared, b - 1.0)
            grad_scalar /= a * pow(dist_squared, b) + 1.0
        else:
            grad_scalar = 0.0

        grad_scalar *= weights[i]
        for d in range(dim):
            pos_grad = clip(grad_scalar * (current[d] - other[d]))
            pos_grads[j][d] += pos_grad

    for i in range(n_vertices):
        # Get necessary data regarding current point and the quadtree cells
        for d in range(dim):
            current[d] = head_embedding[j, d]
        idj = qt.summarize(current, summary, 0.25) # 0.25 = theta^2

        # For each quadtree cell with respect to the current point
        for j in range(idj // offset):
            cell_dist = summary[j * offset + dim]
            cell_size = summary[j * offset + dim + 1]

            if cell_dist > 0.0:
                grad_scalar = 2.0 * b
                grad_scalar /= (0.001 + cell_dist) * (a * pow(cell_dist, b) + 1)
            else:
                continue
            sum_Q += grad_scalar

            grad_scalar *= (1 - average_weight) * cell_size
            for d in range(dim):
                neg_grads[j][d] += grad_scalar * summary[j * offset + d]

    for i in range(n_vertices):
        for j in range(dim):
            all_grads[i][j] = pos_grads[i][j] - neg_grads[i][j] / sum_Q
            head_embedding[i][j] += all_grads[i][j] * alpha

cdef calculate_barnes_hut_tsne(
        np.float32_t[:, :] head_embedding,
        np.float32_t[:, :] tail_embedding,
        np.int32_t[:] head,
        np.int32_t[:] tail,
        np.float32_t[:] weights,
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
        int i, j, iy_1, iy_2
        float cell_size, cell_dist, grad_scalar
        long offset = dim + 2

    # Allocte memory for data structures
    summary = <float*> malloc(sizeof(float) * n_vertices * offset)
    node_1 = <float*> malloc(sizeof(float) * dim)
    node_2 = <float*> malloc(sizeof(float) * dim)
    cdef np.ndarray[double, mode="c", ndim=2] pos_grads = np.zeros([n_vertices, dim])
    cdef np.ndarray[double, mode="c", ndim=2] neg_grads = np.zeros([n_vertices, dim])
    cdef np.ndarray[double, mode="c", ndim=2] all_grads = np.zeros([n_vertices, dim])

    # Get negative force gradients
    for v in range(n_vertices):
        # Get necessary data regarding current point and the quadtree cells
        for d in range(dim):
            node_1[d] = head_embedding[v, d]
        idj = qt.summarize(node_1, summary, 0.25) # 0.25 = theta^2

        # For each cell that pertains to the current point:
        for i_cell in range(idj // offset):
            cell_dist = summary[i_cell * offset + dim]
            cell_size = summary[i_cell * offset + dim + 1]

            # tsne weight calculation:
            qijZ = 1 / (1 + cell_dist)
            sum_Q += cell_size * qijZ
            grad_scalar = cell_size * qijZ * qijZ
            for d in range(dim):
                neg_grads[v][d] += grad_scalar * summary[i_cell * offset + d]

    neg_grads /= sum_Q

    # Get positive force gradients
    for edge in range(epochs_per_sample.shape[0]):
        node_1_index = head[edge]
        node_2_index = tail[edge]
        for d in range(dim):
            node_1[d] = head_embedding[node_1_index, d]
            node_2[d] = tail_embedding[node_2_index, d]

        dist_squared = rdist(node_1, node_2, dim)
        qijZ = 1 / (1 + dist_squared)
        grad_scalar = qijZ * weights[edge]
        for d in range(dim):
            pos_grads[node_1_index][d] += grad_scalar * (node_1[d] - node_2[d])

    for v in range(n_vertices):
        for d in range(dim):
            # Perform tSNE normalizations here
            all_grads[v][d] = pos_grads[v][d] - neg_grads[v][d]
            head_embedding[v][d] += all_grads[v][d] * alpha

def bh_wrapper(
        np.float32_t[:, :] head_embedding,
        np.float32_t[:, :] tail_embedding,
        np.int32_t[:] head,
        np.int32_t[:] tail,
        np.float32_t[:] weights,
        np.float64_t[:] epochs_per_sample,
        float a,
        float b,
        int dim,
        int n_vertices,
        float alpha,
        int umap_flag
):
    cdef _QuadTree qt = _QuadTree(dim, 1)
    qt.build_tree(head_embedding)
    if umap_flag == 0:
        calculate_barnes_hut_tsne(
            head_embedding,
            tail_embedding,
            head,
            tail,
            weights,
            epochs_per_sample,
            qt,
            a,
            b,
            dim,
            n_vertices,
            alpha
        )
    else:
        calculate_barnes_hut_umap(
            head_embedding,
            tail_embedding,
            head,
            tail,
            weights,
            epochs_per_sample,
            qt,
            a,
            b,
            dim,
            n_vertices,
            alpha
        )
