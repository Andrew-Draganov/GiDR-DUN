import numpy as py_np
cimport numpy as np
cimport cython
from libcpp cimport bool
from libc.stdio cimport printf
from libc.stdlib cimport rand
from libc.math cimport sqrt, log, pow
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange, parallel

# FIXME - add flag for faster calculation!
# Replacing pow with fastpow makes mnist run twice as fast
cdef extern from "fastpow.c" nogil:
    double fastpow "fastPow" (double, double)
cdef extern from "fastpow.c" nogil:
    float fmax "my_fmax" (float, float)
cdef extern from "fastpow.c" nogil:
    float fmin "my_fmin" (float, float)

from sklearn.neighbors._quad_tree cimport _QuadTree

np.import_array()

ctypedef np.float32_t DTYPE_FLOAT
ctypedef np.int32_t DTYPE_INT

cdef float clip(float val, float lower, float upper) nogil:
    return fmax(lower, fmin(val, upper))

cdef double rdist(float* x, float* y, int dim) nogil:
    cdef double result = 0.0
    cdef float diff = 0
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result

###################
##### KERNELS #####
###################

@cython.cdivision(True)
cdef float umap_attraction_grad(double dist_squared, float a, double b) nogil:
    cdef float grad_scalar = 0.0
    grad_scalar = -2.0 * a * b * fastpow(dist_squared, b - 1.0)
    grad_scalar /= a * fastpow(dist_squared, b) + 1.0
    return grad_scalar

@cython.cdivision(True)
cdef float umap_repulsion_grad(double dist_squared, float a, double b) nogil:
    cdef float phi_ijZ = 0.0
    phi_ijZ = 2.0 * b
    phi_ijZ /= (0.001 + dist_squared) * (a * fastpow(dist_squared, b) + 1)
    return phi_ijZ

@cython.cdivision(True)
cdef float kernel_function(double dist_squared, float a, double b) nogil:
    return 1 / (1 + a * fastpow(dist_squared, b))

###################
##### WEIGHTS #####
###################

cdef (float, float) umap_repulsive_force(
        float dist_squared,
        float a,
        float b,
        float cell_size,
        float average_weight,
    ) nogil:
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
    ) nogil:
    cdef float kernel = kernel_function(dist_squared, a, b)
    cdef float Z_component = cell_size * kernel
    cdef float repulsive_force = cell_size * kernel * kernel
    return (repulsive_force, Z_component)


cdef float attractive_force_func(
        int normalization,
        float dist_squared,
        float a,
        float b,
        float edge_weight
    ) nogil:
    if normalization == 1:
        edge_force = umap_attraction_grad(dist_squared, a, b)
    else:
        edge_force = kernel_function(dist_squared, a, b)

    return edge_force * edge_weight

cdef (float, float) repulsive_force_func(
        int normalization,
        float dist_squared,
        float a,
        float b,
        float cell_size,
        float average_weight,
    ) nogil:
    if normalization == 1:
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
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[DTYPE_FLOAT, ndim=2] _cy_umap_sampling(
    int normalization,
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
    float alpha,
    np.ndarray[DTYPE_FLOAT, ndim=1] epochs_per_negative_sample,
    np.ndarray[DTYPE_FLOAT, ndim=1] epoch_of_next_negative_sample,
    np.ndarray[DTYPE_FLOAT, ndim=1] epoch_of_next_sample,
    int i_epoch
):
    cdef:
        int i, j, k, d, n_neg_samples
        float attractive_force, repulsive_force
        float dist_squared

    y1 = <float*> malloc(sizeof(float) * dim)
    y2 = <float*> malloc(sizeof(float) * dim)
    cdef float grad_d = 0.0
    cdef (float, float) rep_outputs
    cdef int n_edges = int(epochs_per_sample.shape[0])
    cdef float average_weight = 0.0
    for i in range(weights.shape[0]):
        average_weight += weights[i]
    average_weight = average_weight / n_edges

    for i in range(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= i_epoch:
            # Gets one of the knn in HIGH-DIMENSIONAL SPACE relative to the sample point
            j = head[i]
            k = tail[i]

            if j == k:
                continue
            for d in range(dim):
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
                grad_d = clip(attractive_force * (y1[d] - y2[d]), -4, 4)
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
                    cell_size=1.0,
                    average_weight=0, # Don't scale by weight since we sample according to weight distribution
                )
                repulsive_force = rep_outputs[0]

                for d in range(dim):
                    if repulsive_force > 0.0:
                        grad_d = clip(repulsive_force * (y1[d] - y2[d]), -4, 4)
                    else:
                        grad_d = 4.0
                    head_embedding[j, d] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )
    return forces

def cy_umap_sampling(
    int normalization,
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
    float alpha,
    np.ndarray[DTYPE_FLOAT, ndim=1] epochs_per_negative_sample,
    np.ndarray[DTYPE_FLOAT, ndim=1] epoch_of_next_negative_sample,
    np.ndarray[DTYPE_FLOAT, ndim=1] epoch_of_next_sample,
    int i_epoch
):
    _cy_umap_sampling(
        normalization,
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
    int normalization,
    float [:, :] head_embedding,
    float [:, :] tail_embedding,
    int [:] head,
    int [:] tail,
    float [:] weights,
    float [:, :] forces,
    float [:] epochs_per_sample,
    float a,
    double b,
    int dim,
    int n_vertices,
    float alpha,
):
    cdef:
        int i, j, k, d
        float Z = 0.0
        float [:] Z_array
        float [:, :] attractive_forces
        float [:, :] repulsive_forces
        float attractive_force, repulsive_force
        float dist_squared

    # FIXME - this is faster if zeros is replaced by empty
    cdef int n_edges = int(epochs_per_sample.shape[0])
    Z_array = py_np.zeros([n_edges], dtype=py_np.float32)
    attractive_forces = py_np.zeros([n_vertices, dim], dtype=py_np.float32)
    repulsive_forces = py_np.zeros([n_vertices, dim], dtype=py_np.float32)
    y1 = <float*> malloc(sizeof(float) * dim)
    y2 = <float*> malloc(sizeof(float) * dim)
    cdef float grad_d = 0.0
    cdef (float, float) rep_outputs
    cdef float average_weight = 0.0
    for i in range(weights.shape[0]):
        average_weight += weights[i]
    average_weight /= n_edges

    for i in range(epochs_per_sample.shape[0]):
        # Gets one of the knn in high-dim space relative to the sample point
        j = head[i]
        k = tail[i]
        for d in range(dim):
            y1[d] = head_embedding[j, d]
            y2[d] = tail_embedding[k, d]
        dist_squared = rdist(y1, y2, dim)
        attractive_force = attractive_force_func(
            normalization,
            dist_squared,
            a,
            b,
            weights[i]
        )

        # ANDREW - apply positive force along each nearest neighbor edge
        for d in range(dim):
            grad_d = clip(attractive_force * (y1[d] - y2[d]), -4, 4)
            attractive_forces[j, d] += grad_d
            attractive_forces[k, d] -= grad_d

        # FIXME - add random seed option
        k = rand() % n_vertices
        for d in range(dim):
            y2[d] = tail_embedding[k, d]
        dist_squared = rdist(y1, y2, dim)
        rep_outputs = repulsive_force_func(
            normalization,
            dist_squared,
            a,
            b,
            cell_size=1.0,
            average_weight=average_weight,
        )
        repulsive_force = rep_outputs[0]
        Z_array[i] = rep_outputs[1]

        # Apply repulsive force along random pair of points
        for d in range(dim):
            grad_d = clip(repulsive_force * (y1[d] - y2[d]), -4, 4)
            repulsive_forces[j, d] += grad_d

    for i in range(epochs_per_sample.shape[0]):
        Z += Z_array[i]

    cdef float rep_scalar = -4
    if normalization == 0:
        # avoid division by zero
        rep_scalar = rep_scalar / Z
    cdef float att_scalar = 4

    for v in range(n_vertices):
        for d in range(dim):
            if normalization == 0:
                repulsive_forces[v, d] *= rep_scalar
                attractive_forces[v, d] *= att_scalar
            forces[v, d] = attractive_forces[v, d] + repulsive_forces[v, d]
            head_embedding[v, d] += forces[v, d] * alpha

def cy_umap_uniformly(
    int normalization,
    np.ndarray[DTYPE_FLOAT, ndim=2] _head_embedding,
    np.ndarray[DTYPE_FLOAT, ndim=2] _tail_embedding,
    np.ndarray[DTYPE_INT, ndim=1] _head,
    np.ndarray[DTYPE_INT, ndim=1] _tail,
    np.ndarray[DTYPE_FLOAT, ndim=1] _weights,
    np.ndarray[DTYPE_FLOAT, ndim=2] _forces,
    np.ndarray[DTYPE_FLOAT, ndim=1] _epochs_per_sample,
    float a,
    double b,
    int dim,
    int n_vertices,
    float alpha,
    np.ndarray[DTYPE_FLOAT, ndim=1] _epochs_per_negative_sample,
    np.ndarray[DTYPE_FLOAT, ndim=1] _epoch_of_next_negative_sample,
    np.ndarray[DTYPE_FLOAT, ndim=1] _epoch_of_next_sample,
    int i_epoch
):
    # Convert numpy arrays to cython memoryviews
    cdef float [:, :] head_embedding = _head_embedding
    cdef float [:, :] tail_embedding = _tail_embedding
    cdef int [:] head = _head
    cdef int [:] tail = _tail
    cdef float [:] weights = _weights
    cdef float [:, :] forces = _forces
    cdef float [:] epochs_per_sample = _epochs_per_sample

    _cy_umap_uniformly(
        normalization,
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
        alpha,
    )



##### BARNES-HUT CODE #####

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef np.ndarray[DTYPE_FLOAT, ndim=2] calculate_barnes_hut(
    int normalization,
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
    float alpha,
):
    cdef:
        int i, j, k, l, num_cells, d, i_cell
        float cell_size, cell_dist, grad_scalar, dist_squared
        float theta = 0.5
        long offset = dim + 2
        long dim_index, cell_metadata
        float attractive_force, repulsive_force
        np.ndarray[DTYPE_FLOAT, ndim=2] attractive_forces
        np.ndarray[DTYPE_FLOAT, ndim=2] repulsive_forces
        float [:] Z_array
        float Z = 0.0

    attractive_forces = py_np.zeros([n_vertices, dim], dtype=py_np.float32)
    repulsive_forces = py_np.zeros([n_vertices, dim], dtype=py_np.float32)
    # Allocte memory for data structures
    cell_summaries = <float*> malloc(sizeof(float) * n_vertices * offset)
    y1 = <float*> malloc(sizeof(float) * dim)
    y2 = <float*> malloc(sizeof(float) * dim)
    cdef (float, float) rep_outputs
    # FIXME - what type should these be for optimal performance?

    cdef int n_edges = int(epochs_per_sample.shape[0])
    Z_array = py_np.zeros([n_edges], dtype=py_np.float32)

    cdef float average_weight = 0.0
    for i in range(weights.shape[0]):
        average_weight += weights[i]
    average_weight /= n_edges

    for edge in range(n_edges):
        # Get vertices on either side of the edge
        j = head[edge] # head is the incoming data being transformed
        k = tail[edge] # tail is what we fit to

        for d in range(dim):
            # FIXME - index should be typed for more efficient access
            y1[d] = head_embedding[j, d]
            y2[d] = tail_embedding[k, d]
        dist_squared = rdist(y1, y2, dim)
        attractive_force = attractive_force_func(
            normalization,
            dist_squared,
            a,
            b,
            weights[edge]
        )
        for d in range(dim):
            attractive_forces[j, d] += clip(attractive_force * (y1[d] - y2[d]), -4, 4)

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
            rep_outputs = repulsive_force_func(
                normalization,
                cell_dist,
                a,
                b,
                cell_size,
                average_weight,
            )
            repulsive_force = rep_outputs[0]
            Z_array[i] = rep_outputs[1]
            for d in range(dim):
                dim_index = i_cell * offset + d
                repulsive_forces[v, d] += repulsive_force * cell_summaries[dim_index]

    for i in range(epochs_per_sample.shape[0]):
        Z += Z_array[i]

    cdef float rep_scalar = -4 / Z
    cdef float att_scalar = 4
    for v in range(n_vertices):
        for d in range(dim):
            if normalization == 'tsne':
                repulsive_forces[v, d] *= rep_scalar
                attractive_forces[v, d] *= att_scalar

            forces[v, d] = (attractive_forces[v, d] + repulsive_forces[v, d]) \
                           + 0.9 * forces[v, d]
            head_embedding[v, d] -= forces[v, d] * alpha

    return forces

def bh_wrapper(
    int normalization,
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
    float alpha,
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
        normalization,
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
        alpha
    )

def cy_optimize_layout(
    str optimize_method,
    str _normalization,
    np.ndarray[DTYPE_FLOAT, ndim=2] head_embedding,
    np.ndarray[DTYPE_FLOAT, ndim=2] tail_embedding,
    np.ndarray[DTYPE_INT, ndim=1] head,
    np.ndarray[DTYPE_INT, ndim=1] tail,
    np.ndarray[DTYPE_FLOAT, ndim=1] weights,
    int n_epochs,
    int n_vertices,
    np.ndarray[DTYPE_FLOAT, ndim=1] epochs_per_sample,
    float a,
    double b,
    float alpha,
    float negative_sample_rate,
    verbose=True
):
    cdef:
        int dim, i_epoch
        float initial_alpha
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
    if 'barnes_hut' in optimize_method:
        for i in range(weights.shape[0]):
            weight_sum += weights[i]
        for i in range(weights.shape[0]):
            weights[i] /= weight_sum
        initial_alpha = alpha * 200
    else:
        initial_alpha =  alpha

    cdef int normalization
    if _normalization == 'umap':
        normalization = 1
    else:
        normalization = 0

    single_step_functions = {
        'cy_umap_uniform': cy_umap_uniformly,
        'cy_umap_sampling': cy_umap_sampling,
        'cy_barnes_hut': bh_wrapper
    }
    single_step = single_step_functions[optimize_method]

    for i_epoch in range(n_epochs):
        single_step(
            normalization,
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
            alpha,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            i_epoch
        )
        if verbose and i_epoch % int(n_epochs / 10) == 0:
            print("Completed ", i_epoch, " / ", n_epochs, "epochs")

        alpha = initial_alpha * (1.0 - (float(i_epoch) / float(n_epochs)))
    return head_embedding

