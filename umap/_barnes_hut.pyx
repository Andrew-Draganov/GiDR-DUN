import numpy as np
cimport numpy as np
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


cdef rdist(np.float64_t[:] x, np.float64_t[:] y):
    """Reduced Euclidean distance.

    Parameters
    ----------
    x: array of shape (embedding_dim,)
    y: array of shape (embedding_dim,)

    Returns
    -------
    The squared euclidean distance between x and y
    """
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result


cdef calculate_barnes_hut(
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
        long deg_freedom = 1

    # Allocte memory for data structures
    summary = <float*> malloc(sizeof(float) * n_vertices * offset)
    current = <float*> malloc(sizeof(float) * dim)
    other = <float*> malloc(sizeof(float) * dim)
    cdef np.ndarray[float, mode="c", ndim=2] pos_grads = np.zeros([n_vertices, dim])
    cdef np.ndarray[float, mode="c", ndim=2] neg_grads = np.zeros([n_vertices, dim])
    cdef np.ndarray[float, mode="c", ndim=2] all_grads = np.zeros([n_vertices, dim])

    for i in range(epochs_per_sample.shape[0]):
        # Gets one of the knn in HIGH-DIMENSIONAL SPACE relative to the sample point
        j = head[i]
        k = tail[i]

        # ANDREW - pick random vertex from knn for calculating attractive force
        # t-SNE sums over all knn's attractive forces
        current = head_embedding[j]
        other = tail_embedding[k]
        dist_squared = rdist(current, other)

        # FIXME FIXME use a and b!!!
        if dist_squared > 0.0:
            # ANDREW - this is the actual attractive force for UMAP
            grad_coeff = -2.0 * a * b * pow(dist_squared, b - 1.0)
            grad_coeff /= a * pow(dist_squared, b) + 1.0
        else:
            grad_coeff = 0.0

        grad_coeff *= weights[i]
        for d in range(dim):
            pos_grad = clip(grad_coeff * (current[d] - other[d]))
            pos_grads[j][d] += pos_grad

        # Get necessary data regarding current point and the quadtree cells
        idj = qt.summarize(current, summary, 0.25) # 0.25 = theta^2

        # For each cell that pertains to the current point
        for l in range(idj // offset):
            cell_dist = summary[l * offset + dim]
            cell_size = summary[l * offset + dim + 1]
            qijZ = deg_freedom / (deg_freedom + cell_dist)

            if deg_freedom != 1:
                qijZ = qijZ ** ((deg_freedom + 1) / 2)

            sum_Q += cell_size * qijZ
            grad_scalar = cell_size * qijZ * qijZ
            for d in range(dim):
                neg_grad = grad_scalar * summary[j * offset: j * offset + dim]
                neg_grads[i][d] += neg_grad


    for i in range(n_vertices):
        for j in range(dim):
            all_grads[i][j] = pos_grads[i][j] + neg_grads[i][j] / sum_Q
            head_embedding[i][j] += all_grads[i][j] * alpha

def calc_barnes_hut_wrapper(
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
):
    cdef _QuadTree qt = _QuadTree(dim, 1)
    qt.build_tree(head_embedding)
    calculate_barnes_hut(
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
