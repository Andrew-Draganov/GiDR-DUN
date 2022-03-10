import numpy as py_np
cimport numpy as np
cimport cython
from libc.stdio cimport printf
from libc.math cimport sqrt
from libc.stdlib cimport rand
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange, parallel

from sklearn.neighbors._quad_tree cimport _QuadTree
np.import_array()

cdef extern from "cython_utils.h" nogil:
    float clip(float value, float lower, float upper)
cdef extern from "cython_utils.h" nogil:
    float sq_euc_dist(float* x, float* y, int dim)
cdef extern from "cython_utils.h" nogil:
    float get_lr(float initial_lr, int i_epoch, int n_epochs) 
cdef extern from "cython_utils.h" nogil:
    void print_status(int i_epoch, int n_epochs)
cdef extern from "cython_utils.h" nogil:
    float umap_repulsion_grad(float dist_squared, float a, float b)
cdef extern from "cython_utils.h" nogil:
    float kernel_function(float dist_squared, float a, float b)
cdef extern from "cython_utils.h" nogil:
    float attractive_force_func(
            int normalized,
            float dist_squared,
            float a,
            float b,
            float edge_weight
    )

cdef extern from "cython_utils.h" nogil:
    void repulsive_force_func(
            float* rep_func_outputs,
            int normalized,
            float dist_squared,
            float a,
            float b,
            float cell_size,
            float average_weight
    )


# cdef extern from "gpu_dim_reduction_wrapper.c":
#     void cf()
# 
cdef extern from "optimize_funcs.c" nogil:
    void simple_single_epoch(
        int normalized,
        int sym_attraction,
        int momentum,
        float* head_embedding,
        float* tail_embedding,
        int* head,
        int* tail,
        float* weights,
        long* neighbor_counts,
        float* all_updates,
        float* gains,
        float a,
        float b,
        int dim,
        int n_vertices,
        float lr,
        int i_epoch,
        int n_edges
    )
cdef extern from "optimize_funcs.c" nogil:
    void full_single_epoch(
        int normalized,
        int sym_attraction,
        int momentum,
        float* head_embedding,
        float* tail_embedding,
        int* head,
        int* tail,
        float* weights,
        long* neighbor_counts,
        float* all_updates,
        float* gains,
        float a,
        float b,
        int dim,
        int n_vertices,
        float lr,
        int i_epoch,
        int n_edges
    )

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

ctypedef np.float32_t DTYPE_FLOAT
ctypedef np.int32_t DTYPE_INT

@cython.boundscheck(False)
@cython.cdivision(True)
cdef float get_avg_weight(float[:] weights) nogil:
    cdef float average_weight = 0.0
    for i in range(weights.shape[0]):
        average_weight += weights[i]
    average_weight /= float(weights.shape[0])
    return average_weight

@cython.boundscheck(False)
@cython.cdivision(True)
cdef float _get_avg_weight(float* weights, int n_edges) nogil:
    cdef float average_weight = 0.0
    for i in range(n_edges):
        average_weight += weights[i]
    average_weight /= float(n_edges)
    return average_weight

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _cy_umap_sampling(
    int normalized,
    int sym_attraction,
    float[:, :] head_embedding,
    float[:, :] tail_embedding,
    int[:] head,
    int[:] tail,
    float[:] weights,
    float[:] epochs_per_sample,
    float a,
    float b,
    int dim,
    int n_vertices,
    float lr,
    float[:] epochs_per_negative_sample,
    float[:] epoch_of_next_negative_sample,
    float[:] epoch_of_next_sample,
    int i_epoch,
):
    cdef:
        int i, j, k, n_neg_samples, edge, p
        # Can't reuse loop variables in a with nogil, parallel(): block,
        #   so we create a new variable for each loop
        int v1, v2
        int d1, d2, d3, d4
        float grad_d1, grad_d2
        float attractive_force, repulsive_force
        float dist_squared
        float *y1
        float *y2
        float *rep_func_outputs

    cdef float grad_d = 0.0
    cdef int n_edges = int(epochs_per_sample.shape[0])

    with nogil, parallel():
        y1 = <float*> malloc(sizeof(float) * dim)
        y2 = <float*> malloc(sizeof(float) * dim)
        rep_func_outputs = <float*> malloc(sizeof(float) * 2)
        for i in prange(epochs_per_sample.shape[0]):
            if epoch_of_next_sample[i] <= i_epoch:
                # Gets one of the knn in HIGH-DIMENSIONAL SPACE relative to the sample point
                j = head[i]
                k = tail[i]

                for d1 in range(dim):
                    y1[d1] = head_embedding[j, d1]
                    y2[d1] = tail_embedding[k, d1]
                dist_squared = sq_euc_dist(y1, y2, dim)
                attractive_force = attractive_force_func(
                    normalized,
                    dist_squared,
                    a,
                    b,
                    1
                )

                for d2 in range(dim):
                    grad_d1 = clip(attractive_force * (y1[d2] - y2[d2]), -4, 4)
                    head_embedding[j, d2] -= grad_d1 * lr
                    if sym_attraction:
                        head_embedding[k, d2] += grad_d1 * lr

                epoch_of_next_sample[i] += epochs_per_sample[i]

                # ANDREW - Picks random vertices from ENTIRE graph and calculates repulsive forces
                # FIXME - add random seed option
                n_neg_samples = int(
                    (i_epoch - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
                )
                for p in range(n_neg_samples):
                    k = rand() % n_vertices
                    for d3 in range(dim):
                        y2[d3] = tail_embedding[k, d3]
                    dist_squared = sq_euc_dist(y1, y2, dim)
                    repulsive_force_func(
                        rep_func_outputs,
                        normalized,
                        dist_squared,
                        a,
                        b,
                        1.0,
                        0, # Don't scale by weight since we sample according to weight distribution
                    )
                    repulsive_force = rep_func_outputs[0]

                    for d4 in range(dim):
                        grad_d2 = clip(repulsive_force * (y1[d4] - y2[d4]), -4, 4)
                        head_embedding[j, d4] += grad_d2 * lr

                epoch_of_next_negative_sample[i] += (
                    n_neg_samples * epochs_per_negative_sample[i]
                )
        free(y1)
        free(y2)

def cy_umap_sampling(
    int normalized,
    int sym_attraction,
    int momentum,
    float[:, :] head_embedding,
    float[:, :] tail_embedding,
    int[:] head,
    int[:] tail,
    float[:] weights,
    long[:] neighbor_counts,
    float[:] epochs_per_sample,
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
        np.ndarray[DTYPE_FLOAT, ndim=1] epochs_per_negative_sample,
        np.ndarray[DTYPE_FLOAT, ndim=1] epoch_of_next_negative_sample,
        np.ndarray[DTYPE_FLOAT, ndim=1] epoch_of_next_sample,

    epochs_per_negative_sample = py_np.zeros_like(epochs_per_sample)
    epoch_of_next_negative_sample = py_np.zeros_like(epochs_per_sample)
    epoch_of_next_sample = py_np.zeros_like(epochs_per_sample)
    # ANDREW - perform negative samples x times more often
    #          by making the number of epochs between samples smaller
    for i in range(weights.shape[0]):
        epochs_per_negative_sample[i] = epochs_per_sample[i] / negative_sample_rate
        epoch_of_next_negative_sample[i] = epochs_per_negative_sample[i]
        epoch_of_next_sample[i] = epochs_per_sample[i]

    for i_epoch in range(n_epochs):
        lr = get_lr(initial_lr, i_epoch, n_epochs)
        _cy_umap_sampling(
            normalized,
            sym_attraction,
            head_embedding,
            tail_embedding,
            head,
            tail,
            weights,
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

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef void get_kernels(
#     float *local_Z,
#     float *attr_forces,
#     float *rep_forces,
#     float *attr_vecs,
#     float *rep_vecs,
#     int[:] head,
#     int[:] tail,
#     float[:, :] head_embedding,
#     float[:, :] tail_embedding,
#     float[:] weights,
#     int normalized,
#     int n_vertices,
#     int n_edges,
#     int i_epoch,
#     int dim,
#     float a,
#     float b,
#     float average_weight
# ) nogil:
#     cdef:
#         int edge, j, k, d
#         float dist_squared, weight_scalar
#         float *y1
#         float *y2
#         float *rep_func_outputs
# 
#     with parallel():
#         local_Z[0] = 0
#         y1 = <float*> malloc(sizeof(float) * dim)
#         y2 = <float*> malloc(sizeof(float) * dim)
#         rep_func_outputs = <float*> malloc(sizeof(float) * 2)
#         for edge in prange(n_edges):
#             j = head[edge]
#             k = tail[edge]
#             for d in range(dim):
#                 y1[d] = head_embedding[j, d]
#                 y2[d] = tail_embedding[k, d]
#                 attr_vecs[edge * dim + d] = y1[d] - y2[d]
#             dist_squared = sq_euc_dist(y1, y2, dim)
# 
#             # t-SNE early exaggeration
#             if i_epoch < 100:
#                 weight_scalar = 4
#             else:
#                 weight_scalar = 1
# 
#             attr_forces[edge] = attractive_force_func(
#                 normalized,
#                 dist_squared,
#                 a,
#                 b,
#                 weights[edge] * weight_scalar
#             )
# 
#             k = rand() % n_vertices
#             for d in range(dim):
#                 y2[d] = tail_embedding[k, d]
#                 rep_vecs[edge * dim + d] = y1[d] - y2[d]
#             dist_squared = sq_euc_dist(y1, y2, dim)
#             repulsive_force_func(
#                 rep_func_outputs,
#                 normalized,
#                 dist_squared,
#                 a,
#                 b,
#                 1.0,
#                 average_weight,
#             )
#             rep_forces[edge] = rep_func_outputs[0]
#             local_Z[0] += rep_func_outputs[1]
# 
#         free(rep_func_outputs)
#         free(y1)
#         free(y2)
# 
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef void gather_gradients(
#     float *local_attr_grads,
#     float *local_rep_grads,
#     int[:] head,
#     int[:] tail,
#     float* attr_forces,
#     float* rep_forces,
#     float* attr_vecs,
#     float* rep_vecs,
#     int sym_attraction,
#     int n_vertices,
#     int n_edges,
#     int dim,
#     float Z
# ) nogil:
#     cdef:
#         int j, k, v, d, edge, index
#         float grad_d
# 
#     with parallel():
#         # Fill allocated memory with zeros
#         for v in prange(n_vertices):
#             for d in range(dim):
#                 index = v * dim + d
#                 local_attr_grads[index] = 0
#                 local_rep_grads[index] = 0
# 
#         for edge in prange(n_edges):
#             j = head[edge]
#             for d in range(dim):
#                 grad_d = clip(attr_forces[edge] * attr_vecs[edge * dim + d], -4, 4)
#                 local_attr_grads[j * dim + d] += grad_d
#                 if sym_attraction:
#                     k = tail[edge]
#                     local_attr_grads[k * dim + d] -= grad_d
# 
#             for d in range(dim):
#                 grad_d = clip(rep_forces[edge] * rep_vecs[edge * dim + d], -4, 4)
#                 local_rep_grads[j * dim + d] += grad_d / Z


# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(True)
# cdef void _cy_umap_uniformly(
#     int normalized,
#     int sym_attraction,
#     int momentum,
#     float* head_embedding,
#     float* tail_embedding,
#     int* head,
#     int* tail,
#     float* weights,
#     float* all_updates,
#     float* gains,
#     float a,
#     float b,
#     int dim,
#     int n_vertices,
#     float lr,
#     int i_epoch,
#     int n_edges
# ):
#     cdef:
#         int i, j, k, d, v, index, edge
#         float *attr_grads
#         float *rep_grads
#
#         float *attr_vecs
#         float *rep_vecs
#         float *attr_forces
#         float *rep_forces
#
#         float Z
#         float scalar
#
#     attr_forces = <float*> malloc(sizeof(float) * n_edges)
#     rep_forces = <float*> malloc(sizeof(float) * n_edges)
#     attr_vecs = <float*> malloc(sizeof(float) * n_edges * dim)
#     rep_vecs = <float*> malloc(sizeof(float) * n_edges * dim)
#
#     attr_grads = <float*> malloc(sizeof(float) * n_vertices * dim)
#     rep_grads = <float*> malloc(sizeof(float) * n_vertices * dim)
#
#     cdef float grad = 0.0
#     cdef float grad_d = 0.0
#     cdef float average_weight = _get_avg_weight(weights, n_edges)
#
#     with nogil, parallel():
#         for v in prange(n_vertices):
#             for d in range(dim):
#                 index = v * dim + d
#                 attr_grads[index] = 0
#                 rep_grads[index] = 0
#
#     with nogil:
#         Z = get_kernels(
#             attr_forces,
#             rep_forces,
#             attr_vecs,
#             rep_vecs,
#             head,
#             tail,
#             head_embedding,
#             tail_embedding,
#             weights,
#             normalized,
#             n_vertices,
#             n_edges,
#             i_epoch,
#             dim,
#             a,
#             b,
#             average_weight
#         )
#         if not normalized:
#             Z = 1
#
#         gather_gradients(
#             attr_grads,
#             rep_grads,
#             head,
#             tail,
#             attr_forces,
#             rep_forces,
#             attr_vecs,
#             rep_vecs,
#             sym_attraction,
#             n_vertices,
#             n_edges,
#             dim,
#             Z
#         )
#         free(attr_forces)
#         free(rep_forces)
#         free(attr_vecs)
#         free(rep_vecs)
#
#         for v in range(n_vertices):
#             for d in range(dim):
#                 index = v * dim + d
#                 # Would like to put rep_grads[i] - attr_grads[i] into variable
#                 #   but get cython error "Cannot read reduction variable in loop body"
#                 # So calculate it twice to avoid the error
#
#                 if (rep_grads[index] - attr_grads[index]) * all_updates[index] > 0.0:
#                     gains[index] += 0.2
#                 else:
#                     gains[index] *= 0.8
#                 gains[index] = clip(gains[index], 0.01, 100)
#                 grad_d = (rep_grads[index] - attr_grads[index]) * gains[index]
#
#                 all_updates[index] = grad_d * lr + momentum * 0.9 * all_updates[index]
#                 head_embedding[index] += all_updates[index]
#
#     free(attr_grads)
#     free(rep_grads)


def cy_umap_uniformly(
    int normalized,
    int sym_attraction,
    int momentum,
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] head_embedding,
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] tail_embedding,
    np.ndarray[int, ndim=1, mode='c'] head,
    np.ndarray[int, ndim=1, mode='c'] tail,
    np.ndarray[DTYPE_FLOAT, ndim=1, mode='c'] weights,
    np.ndarray[long, ndim=1, mode='c'] neighbor_counts,
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
        int v, d, index, edge, current_point, next_point
        int n_edges = int(epochs_per_sample.shape[0])
        float *all_updates
        float *gains

    all_updates = <float*> malloc(sizeof(float) * n_vertices * dim)
    gains = <float*> malloc(sizeof(float) * n_vertices * dim)
    for v in range(n_vertices):
        for d in range(dim):
            index = v * dim + d
            all_updates[index] = 0
            gains[index] = 1

    for i_epoch in range(n_epochs):
        lr = get_lr(initial_lr, i_epoch, n_epochs)
        simple_single_epoch(
            normalized,
            sym_attraction,
            momentum,
            &head_embedding[0, 0], # Move from numpy to c pointer arrays
            &tail_embedding[0, 0],
            &head[0],
            &tail[0],
            &weights[0],
            &neighbor_counts[0],
            all_updates,
            gains,
            a,
            b,
            dim,
            n_vertices,
            lr,
            i_epoch,
            n_edges
        )
        if verbose:
            print_status(i_epoch, n_epochs)

    free(all_updates)
    free(gains)
    return head_embedding


##### BARNES-HUT CODE #####

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void bh_collect_attr_grads(
    float *local_attr_grads,
    int[:] head,
    int[:] tail,
    float[:, :] head_embedding,
    float[:, :] tail_embedding,
    float[:] weights,
    int normalized,
    int sym_attraction,
    int n_vertices,
    int n_edges,
    int i_epoch,
    int dim,
    float a,
    float b,
) nogil:
    cdef:
        int edge, j, k, d, v
        float dist_squared, weight_scalar, grad_d
        float *y1
        float *y2
        float attr_force

    for v in range(n_vertices):
        for d in range(dim):
            local_attr_grads[v * dim + d] = 0

    with parallel():
        y1 = <float*> malloc(sizeof(float) * dim)
        y2 = <float*> malloc(sizeof(float) * dim)
        for edge in prange(n_edges):
            j = head[edge]
            k = tail[edge]
            for d in range(dim):
                y1[d] = head_embedding[j, d]
                y2[d] = tail_embedding[k, d]
            dist_squared = sq_euc_dist(y1, y2, dim)

            # t-SNE early exaggeration
            if i_epoch < 100:
                weight_scalar = 4
            else:
                weight_scalar = 1

            attr_force = attractive_force_func(
                normalized,
                dist_squared,
                a,
                b,
                weights[edge] * weight_scalar
            )

            for d in range(dim):
                grad_d = attr_force * (y1[d] - y2[d])
                local_attr_grads[j * dim + d] += grad_d
                if sym_attraction:
                    local_attr_grads[k * dim + d] -= grad_d
        free(y1)
        free(y2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void bh_collect_rep_grads(
    float *local_Z,
    float *local_rep_grads,
    float[:, :] head_embedding,
    _QuadTree qt,
    int normalized,
    int n_vertices,
    int dim,
    float a,
    float b,
    float average_weight
) nogil:
    cdef:
        int v, i_cell, j, k, d
        int num_cells, dim_index
        long cell_metadata
        float theta = 0.5
        long offset = dim + 2
        float dist_squared, cell_dist, cell_size, grad_d
        float *cell_summaries
        float *rep_func_outputs
        float *y1

    for v in range(n_vertices):
        for d in range(dim):
            local_rep_grads[v * dim + d] = 0

    local_Z[0] = 0
    with parallel():
        cell_summaries = <float*> malloc(sizeof(float) * n_vertices * offset)
        rep_func_outputs = <float*> malloc(sizeof(float) * 2)
        y1 = <float*> malloc(sizeof(float) * dim)
        for v in prange(n_vertices):
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
                # Ignoring small cells gives clusters that REALLY emphasize
                #      local relationships while generally maintaining global ones
                # if cell_size < 3:
                #     continue
                repulsive_force_func(
                    rep_func_outputs,
                    normalized,
                    cell_dist,
                    a,
                    b,
                    cell_size,
                    average_weight,
                )
                local_Z[0] += rep_func_outputs[1]

                for d in range(dim):
                    dim_index = i_cell * offset + d
                    grad_d = rep_func_outputs[0] * cell_summaries[dim_index]
                    local_rep_grads[v * dim + d] += grad_d

        free(cell_summaries)
        free(y1)
        free(rep_func_outputs)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void calculate_barnes_hut(
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
    _QuadTree qt,
    float a,
    float b,
    int dim,
    int n_vertices,
    float lr,
    int i_epoch
):
    cdef:
        int i, j, k, l, num_cells, d, i_cell, v
        float grad_scalar, dist_squared, grad_d, Z
        float theta = 0.5
        long dim_index, index
        float *all_attr_grads
        float *all_rep_grads
        float *local_attr_grads
        float *local_rep_grads
        float *local_Z

    cdef int n_edges = int(head.shape[0])
    cdef float scalar = 4 * a * b
    cdef float average_weight = get_avg_weight(weights)

    # Allocte memory for data structures
    all_attr_grads = <float*> malloc(sizeof(float) * n_vertices * dim)
    all_rep_grads = <float*> malloc(sizeof(float) * n_vertices * dim)
    local_attr_grads = <float*> malloc(sizeof(float) * n_vertices * dim)
    local_rep_grads = <float*> malloc(sizeof(float) * n_vertices * dim)
    local_Z = <float*> malloc(sizeof(float))
    for v in range(n_vertices):
        for d in range(dim):
            index = v * dim + d
            all_attr_grads[index] = 0
            all_rep_grads[index] = 0

    with nogil:
        Z = 0
        bh_collect_attr_grads(
            local_attr_grads,
            head,
            tail,
            head_embedding,
            tail_embedding,
            weights,
            normalized,
            sym_attraction,
            n_vertices,
            n_edges,
            i_epoch,
            dim,
            a,
            b,
        )

        bh_collect_rep_grads(
            local_Z,
            local_rep_grads,
            head_embedding,
            qt,
            normalized,
            n_vertices,
            dim,
            a,
            b,
            average_weight
        )
        Z += local_Z[0]
        if not normalized:
            Z = 1
            # Dividing by n_vertices means we have the same one-to-one
            #   relationship between attractive and repulsive forces
            #   as in traditional UMAP
            # FIXME - why does this create a perfect circle???
            scalar /= n_vertices

        with parallel():
            for v in prange(n_vertices):
                for d in range(dim):
                    index = v * dim + d
                    all_attr_grads[index] += local_attr_grads[index] * scalar
                    all_rep_grads[index] += local_rep_grads[index] * scalar / Z

        with parallel():
            for v in prange(n_vertices):
                for d in range(dim):
                    index = v * dim + d
                    if (all_rep_grads[index] - all_attr_grads[index]) * all_updates[index] > 0.0:
                        gains[index] += 0.2
                    else:
                        gains[index] *= 0.8
                    gains[index] = clip(gains[index], 0.01, 1000)
                    grad_d = (all_rep_grads[index] - all_attr_grads[index]) * gains[index]

                    if momentum:
                        all_updates[index] = grad_d * lr + 0.9 * all_updates[index]
                    else:
                        all_updates[index] = grad_d * lr

                    head_embedding[v, d] += all_updates[index]

    free(local_Z)

    free(local_attr_grads)
    free(local_rep_grads)

    free(all_attr_grads)
    free(all_rep_grads)


def bh_wrapper(
    int normalized,
    int sym_attraction,
    int momentum,
    float[:, :] head_embedding,
    float[:, :] tail_embedding,
    int[:] head,
    int[:] tail,
    float[:] weights,
    long[:] neighbor_counts,
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
    """
    Wrapper to call barnes_hut optimization
    Require a regular def function to call from python file
    But this standard function in a .pyx file can call the cdef function
    """
    # Can only define cython quadtree in a cython function
    cdef:
        float *all_updates
        float *gains
    cdef _QuadTree qt = _QuadTree(dim, 1)
    all_updates = <float*> malloc(sizeof(float) * n_vertices * dim)
    gains = <float*> malloc(sizeof(float) * n_vertices * dim)
    for v in range(n_vertices):
        for d in range(dim):
            index = v * dim + d
            all_updates[index] = 0
            gains[index] = 1

    for i_epoch in range(n_epochs):
        qt.build_tree(head_embedding)
        lr = get_lr(initial_lr, i_epoch, n_epochs)
        calculate_barnes_hut(
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
            qt,
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

    return head_embedding

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def cy_optimize_layout(
    str optimize_method,
    int normalized,
    int sym_attraction,
    int momentum,
    np.ndarray[DTYPE_INT, ndim=1, mode='c'] head,
    np.ndarray[DTYPE_INT, ndim=1, mode='c'] tail,
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] head_embedding,
    np.ndarray[DTYPE_FLOAT, ndim=2, mode='c'] tail_embedding,
    np.ndarray[DTYPE_FLOAT, ndim=1, mode='c'] weights,
    np.ndarray[long, ndim=1, mode='c'] neighbor_counts,
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
        int i_epoch, n_edges
        cdef int dim = head_embedding.shape[1]

    # Perform weight scaling on high-dimensional relationships
    cdef float weight_sum = 0.0
    if normalized:
        for i in range(weights.shape[0]):
            weight_sum += weights[i]
        for i in range(weights.shape[0]):
            weights[i] /= weight_sum
        initial_lr *= 200

    optimize_dict = {
        'cy_umap_uniform': cy_umap_uniformly,
        'cy_umap_sampling': cy_umap_sampling,
        'cy_barnes_hut': bh_wrapper
    }
    optimize_fn = optimize_dict[optimize_method]

    head_embedding = optimize_fn(
        normalized,
        sym_attraction,
        momentum,
        head_embedding,
        tail_embedding,
        head,
        tail,
        weights,
        neighbor_counts,
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
