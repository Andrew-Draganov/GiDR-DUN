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

cdef float repel_clip(float val, float scalar, float lower, float upper) nogil:
    if val > 0.0:
        return clip(val * scalar, lower, upper)
    return 4.0

cdef float euc_dist(float* x, float* y, int dim):
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

###################
##### KERNELS #####
###################

@cython.cdivision(True)
cdef float umap_attraction_grad(float dist_squared, float a, float b):
    cdef float grad_scalar = 0.0
    grad_scalar = -2.0 * a * b * fastpow(dist_squared, b - 1.0)
    grad_scalar /= a * fastpow(dist_squared, b) + 1.0
    return grad_scalar

@cython.cdivision(True)
cdef float umap_repulsion_grad(float dist_squared, float a, float b):
    cdef float phi_ijZ = 0.0
    phi_ijZ = 2.0 * b
    phi_ijZ /= (0.001 + dist_squared) * (a * fastpow(dist_squared, b) + 1)
    return phi_ijZ

@cython.cdivision(True)
cdef float kernel_function(float dist_squared, float a, float b):
    if b <= 1:
        return 1 / (1 + a * fastpow(dist_squared, b))
    return fastpow(dist_squared, b - 1) / (1 + a * fastpow(dist_squared, b))

###################
##### WEIGHTS #####
###################

cdef (float, float) umap_repulsive_force(
        float dist_squared,
        float a,
        float b,
        float cell_size,
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
        float cell_size,
        float Z
    ):
    cdef float kernel = kernel_function(dist_squared, a, b)
    Z += cell_size * kernel # Collect the q_ij's contributions into Z
    cdef float repulsive_force = cell_size * kernel * kernel
    return (repulsive_force, Z)

cdef float attractive_force_func(
        int normalized,
        float dist_squared,
        float a,
        float b,
        float edge_weight
    ):
    if not normalized:
        edge_force = umap_attraction_grad(dist_squared, a, b)
    else:
        edge_force = kernel_function(dist_squared, a, b)

    return edge_force * edge_weight

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef (float*, float) multiple_normed_repulsions(
        float* base_point,
        float* points,
        int num_points,
        int dim
    ):
    cdef int i, j
    repulsive_force = <float*> malloc(sizeof(float) * dim)
    for d in range(dim):
        repulsive_force[d] = 0
    current = <float*> malloc(sizeof(float) * dim)
    dists = <float*> malloc(sizeof(float) * num_points)
    kernels = <float*> malloc(sizeof(float) * num_points)
    sq_kernels = <float*> malloc(sizeof(float) * num_points)
    for i in range(num_points):
        for j in range(dim):
            current[j] = points[dim * i + j]
        dists[i] = euc_dist(base_point, current, dim)
        kernels[i] = (1 + dists[i])
        sq_kernels[i] = kernels[i] * kernels[i]

    cdef float kernel_sum = 0
    cdef float kernel_prod = 1
    cdef float Z = 0
    for i in range(num_points):
        kernel_sum += 1 / sq_kernels[i]
        kernel_prod *= sq_kernels[i]

    for d in range(dim):
        for i in range(num_points):
            for j in range(dim):
                current[j] = points[dim * i + j]
            repulsive_force[d] -= current[d] * kernel_prod / sq_kernels[i]
            Z += 1 / kernels[i]
        repulsive_force[d] /= kernel_prod
        repulsive_force[d] += base_point[d] * kernel_sum

    Z *= 3
    return repulsive_force, Z

cdef (float, float) repulsive_force_func(
        int normalized,
        float dist_squared,
        float a,
        float b,
        float cell_size,
        float average_weight,
        float Z
    ):
    if not normalized:
        return umap_repulsive_force(
            dist_squared,
            a,
            b,
            cell_size,
            average_weight
        )
    return tsne_repulsive_force(
        dist_squared,
        a,
        b,
        cell_size,
        Z
    )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _cy_umap_sampling(
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
    int i_epoch
):
    cdef:
        int i, j, k, d, n_neg_samples
        float attractive_force, repulsive_force
        float dist_squared
        float Z = 0.0

    # FIXME FIXME FIXME -- this version has weird localized clusters compared
    # to numba version
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

            for d in range(dim):
                y1[d] = head_embedding[j, d]
                y2[d] = tail_embedding[k, d]
            # ANDREW - optimize positive force for each edge
            dist_squared = euc_dist(y1, y2, dim)
            attractive_force = attractive_force_func(
                normalized,
                dist_squared,
                a,
                b,
                weights[i]
            )

            for d in range(dim):
                grad_d = clip(attractive_force * (y1[d] - y2[d]), -4, 4)
                head_embedding[j, d] += grad_d * lr
                if sym_attraction:
                    head_embedding[k, d] -= grad_d * lr

            epoch_of_next_sample[i] += epochs_per_sample[i]

            # ANDREW - Picks random vertex from ENTIRE graph and calculates repulsive force
            # FIXME - add random seed option
            n_neg_samples = int(
                (i_epoch - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )
            for p in range(n_neg_samples):
                k = rand() % n_vertices
                for d in range(dim):
                    y2[d] = tail_embedding[k, d]
                dist_squared = euc_dist(y1, y2, dim)
                rep_outputs = repulsive_force_func(
                    normalized,
                    dist_squared,
                    a,
                    b,
                    cell_size=1.0,
                    average_weight=0, # Don't scale by weight since we sample according to weight distribution
                    Z=Z
                )
                repulsive_force = rep_outputs[0]
                Z = rep_outputs[1]

                for d in range(dim):
                    grad_d = clip(repulsive_force * (y1[d] - y2[d]), -4, 4)
                    head_embedding[j, d] += grad_d * lr

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )

def cy_umap_sampling(
    int normalized,
    int sym_attraction,
    int momentum,
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
    int i_epoch
):
    _cy_umap_sampling(
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
        i_epoch
    )

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _cy_umap_uniformly(
    int normalized,
    int sym_attraction,
    int momentum,
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
    int i_epoch
):
    cdef:
        int i, j, k, d, v
        np.ndarray[DTYPE_FLOAT, ndim=2] attractive_forces
        np.ndarray[DTYPE_FLOAT, ndim=2] repulsive_forces
        float Z = 0.0
        float theta = 0.5
        float attractive_force, repulsive_force, dist_squared, edge_weight

    attractive_forces = py_np.zeros([n_vertices, dim], dtype=py_np.float32)
    repulsive_forces = py_np.zeros([n_vertices, dim], dtype=py_np.float32)
    y1 = <float*> malloc(sizeof(float) * dim)
    y2 = <float*> malloc(sizeof(float) * dim)

    num_repels = 5
    repel_points = <float*> malloc(sizeof(float) * num_repels * dim)
    grad = <float*> malloc(sizeof(float) * dim)
    cdef int p = 0
    cdef float z_component = 0.0

    cdef float grad_d = 0.0
    cdef (float, float) rep_outputs
    cdef int n_edges = int(epochs_per_sample.shape[0])
    cdef float _average_weight = 0.0
    for i in range(weights.shape[0]):
        _average_weight += weights[i]
    _average_weight /= n_edges

    for edge in range(n_edges):
        # Gets one of the knn in HIGH-DIMENSIONAL SPACE relative to the sample point
        j = head[edge]
        k = tail[edge]
        for d in range(dim):
            y1[d] = head_embedding[j, d]
            y2[d] = tail_embedding[k, d]
        dist_squared = euc_dist(y1, y2, dim)

        # t-SNE early exaggeration
        if i_epoch < 50 and normalized:
            edge_weight = weights[edge] * 4
            average_weight = _average_weight * 4
        else:
            edge_weight = weights[edge]
            average_weight = _average_weight

        # Optimize positive force for each edge
        attractive_force = attractive_force_func(
            normalized,
            dist_squared,
            a,
            b,
            weights[edge]
        )
        for d in range(dim):
            grad_d = clip(attractive_force * (y1[d] - y2[d]), -4, 4)
            attractive_forces[j, d] += grad_d
            if sym_attraction:
                attractive_forces[k, d] -= grad_d

        # Picks random vertex from ENTIRE graph and calculates repulsive force
        # FIXME - add random seed option
        for p in range(num_repels):
            k = rand() % n_vertices
            for d in range(dim):
                repel_points[p * dim + d] = tail_embedding[k, d]
        grad, z_component = multiple_normed_repulsions(
            y1,
            repel_points,
            num_repels,
            dim
        )
        Z += z_component

        # k = rand() % n_vertices
        # for d in range(dim):
        #     y2[d] = tail_embedding[k, d]
        # dist_squared = euc_dist(y1, y2, dim)

        # rep_outputs = repulsive_force_func(
        #     normalized,
        #     dist_squared,
        #     a,
        #     b,
        #     cell_size=1.0,
        #     average_weight=average_weight,
        #     Z=Z
        # )
        # repulsive_force = rep_outputs[0]
        # Z = rep_outputs[1]

        for d in range(dim):
            grad_d = clip(grad[d], -4, 4)
            # repulsive_forces[j, d] += grad_d
            # repulsive_forces[k, d] -= grad_d

    cdef float rep_scalar = 4 * a * b
    cdef float att_scalar = -4 * a * b
    if normalized:
        # avoid division by zero
        rep_scalar /= Z

    for v in range(n_vertices):
        for d in range(dim):
            if normalized:
                repulsive_forces[v, d] = repulsive_forces[v, d] * rep_scalar
                attractive_forces[v, d] = attractive_forces[v, d] * att_scalar
            if momentum == 1:
                forces[v, d] = (attractive_forces[v, d] + repulsive_forces[v, d]) + 0.9 * forces[v, d]
            else:
                forces[v, d] = attractive_forces[v, d] + repulsive_forces[v, d]
            head_embedding[v, d] += forces[v, d] * lr


def cy_umap_uniformly(
    int normalized,
    int sym_attraction,
    int momentum,
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
    int i_epoch
):
    _cy_umap_uniformly(
        normalized,
        sym_attraction,
        momentum,
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
        i_epoch
    )


##### BARNES-HUT CODE #####

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void calculate_barnes_hut(
    int normalized,
    int sym_attraction,
    int momentum,
    np.ndarray[DTYPE_FLOAT, ndim=2] head_embedding,
    np.ndarray[DTYPE_FLOAT, ndim=2] tail_embedding,
    np.ndarray[DTYPE_INT, ndim=1] head,
    np.ndarray[DTYPE_INT, ndim=1] tail,
    np.ndarray[DTYPE_FLOAT, ndim=1] weights,
    np.ndarray[DTYPE_FLOAT, ndim=2] forces,
    np.ndarray[DTYPE_FLOAT, ndim=1] epochs_per_sample,
    _QuadTree qt,
    float a,
    float b,
    int dim,
    int n_vertices,
    float lr,
):
    cdef:
        double Z = 0.0
        int i, j, k, l, num_cells, d, i_cell
        float cell_size, cell_dist, grad_scalar, dist_squared
        float theta = 0.5
        long offset = dim + 2
        long dim_index, cell_metadata
        float attractive_force, repulsive_force
        np.ndarray[DTYPE_FLOAT, ndim=2] attractive_forces
        np.ndarray[DTYPE_FLOAT, ndim=2] repulsive_forces

    # FIXME - add early exaggeration
    attractive_forces = py_np.zeros([n_vertices, dim], dtype=py_np.float32)
    repulsive_forces = py_np.zeros([n_vertices, dim], dtype=py_np.float32)
    # Allocte memory for data structures
    cell_summaries = <float*> malloc(sizeof(float) * n_vertices * offset)
    y1 = <float*> malloc(sizeof(float) * dim)
    y2 = <float*> malloc(sizeof(float) * dim)
    grad = <float*> malloc(sizeof(float) * dim)

    cdef int n_edges = int(epochs_per_sample.shape[0])
    cdef float average_weight = 0.0
    for i in range(weights.shape[0]):
        average_weight += weights[i]
    average_weight /= n_edges

    for edge in range(n_edges):
        # Get vertices on either side of the edge
        j = head[edge] # head is the incoming data being transformed
        k = tail[edge] # tail is what we fit to

        for d in range(dim):
            y1[d] = head_embedding[j, d]
            y2[d] = tail_embedding[k, d]
        dist_squared = euc_dist(y1, y2, dim)
        attractive_force = attractive_force_func(
            normalized,
            dist_squared,
            a,
            b,
            weights[edge]
        )
        for d in range(dim):
            grad[d] = clip(attractive_force * (y1[d] - y2[d]), -4, 4)
            attractive_forces[j, d] += grad[d]
            if sym_attraction:
                attractive_forces[k, d] -= grad[d]

    for v in range(n_vertices):
        # Get necessary data regarding current point and the quadtree cells
        for d in range(dim):
            y1[d] = head_embedding[v, d]
        cell_metadata = qt.summarize(y1, cell_summaries, theta * theta)
        num_cells = cell_metadata // offset

        # For each quadtree cell with respect to the current point
        for i_cell in range(num_cells):
            cell_dist = cell_summaries[i_cell * offset + dim]
            cell_size = cell_summaries[i_cell * offset + dim + 1]
            # FIXME - think more about this cell_size bounding
            # Ignoring small cells gives clusters that REALLY preserve 
            #      local relationships while generally maintaining global ones
            # if cell_size < 1:
            #     continue
            repulsive_force, Z = repulsive_force_func(
                normalized,
                cell_dist,
                a,
                b,
                cell_size,
                average_weight,
                Z
            )
            for d in range(dim):
                dim_index = i_cell * offset + d
                repulsive_forces[v, d] += clip(
                    repulsive_force * cell_summaries[dim_index], -4, 4
                )

    cdef float rep_scalar = 4 / Z
    cdef float att_scalar = -4
    for v in range(n_vertices):
        for d in range(dim):
            if normalized:
                repulsive_forces[v, d] = repulsive_forces[v, d] * rep_scalar
                attractive_forces[v, d] = attractive_forces[v, d] * att_scalar

            if momentum:
                forces[v, d] = attractive_forces[v, d] + repulsive_forces[v, d] \
                               + 0.9 * forces[v, d]
            else:
                forces[v, d] = attractive_forces[v, d] + repulsive_forces[v, d]
            head_embedding[v, d] += forces[v, d] * lr

def bh_wrapper(
    int normalized,
    int sym_attraction,
    int momentum,
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
    int i_epoch
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
        normalized,
        sym_attraction,
        momentum,
        head_embedding,
        tail_embedding,
        head,
        tail,
        weights,
        forces,
        epochs_per_sample,
        qt,
        a,
        b,
        dim,
        n_vertices,
        lr
    )

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def cy_optimize_layout(
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
        np.ndarray[DTYPE_FLOAT, ndim=1] epochs_per_negative_sample,
        np.ndarray[DTYPE_FLOAT, ndim=1] epoch_of_next_negative_sample,
        np.ndarray[DTYPE_FLOAT, ndim=1] epoch_of_next_sample,

    dim = head_embedding.shape[1]
    forces = py_np.zeros([n_vertices, dim], dtype=py_np.float32)

    # ANDREW - perform negative samples x times more often
    #          by making the number of epochs between samples smaller
    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    # Make copies of these two
    epoch_of_next_negative_sample = py_np.ones_like(epochs_per_negative_sample) * epochs_per_negative_sample
    epoch_of_next_sample = py_np.ones_like(epochs_per_sample) * epochs_per_sample

    # Perform weight scaling on high-dimensional relationships
    cdef float weight_sum = 0.0
    if normalized:
        for i in range(weights.shape[0]):
            weight_sum = weight_sum + weights[i]
        for i in range(weights.shape[0]):
            weights[i] = weights[i] / weight_sum
        initial_lr *= 200

    single_step_functions = {
        'cy_umap_uniform': cy_umap_uniformly,
        'cy_umap_sampling': cy_umap_sampling,
        'cy_barnes_hut': bh_wrapper
    }
    single_step = single_step_functions[optimize_method]

    for i_epoch in range(n_epochs):
        lr = initial_lr * (1.0 - (float(i_epoch) / float(n_epochs)))
        single_step(
            normalized,
            sym_attraction,
            momentum,
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
            i_epoch
        )
        if verbose and i_epoch % int(n_epochs / 10) == 0:
            print("Completed ", i_epoch, " / ", n_epochs, "epochs")

    return head_embedding

