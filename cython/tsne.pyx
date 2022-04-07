import numpy as py_np
cimport numpy as np
cimport cython
from libc.stdio cimport printf
from libc.math cimport sqrt, fabs
from libc.stdlib cimport rand
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange, parallel
from sklearn.neighbors._quad_tree cimport _QuadTree
np.import_array()

cdef extern from "cython_utils.c" nogil:
    float clip(float value, float lower, float upper)
cdef extern from "cython_utils.c" nogil:
    float sq_euc_dist(float* x, float* y, int dim)
cdef extern from "cython_utils.c" nogil:
    float ang_dist(float* x, float* y, int dim)
cdef extern from "cython_utils.c" nogil:
    float get_lr(float initial_lr, int i_epoch, int n_epochs, int amplify_grads) 
cdef extern from "cython_utils.c" nogil:
    void print_status(int i_epoch, int n_epochs)
cdef extern from "cython_utils.c" nogil:
    float umap_repulsion_grad(float dist, float a, float b)
cdef extern from "cython_utils.c" nogil:
    float kernel_function(float dist, float a, float b)
cdef extern from "cython_utils.c" nogil:
    float get_avg_weight(float* weights, int n_edges)
cdef extern from "cython_utils.c" nogil:
    float attractive_force_func(
            int normalized,
            int frob,
            float dist,
            float a,
            float b,
            float edge_weight
    )
cdef extern from "cython_utils.c" nogil:
    void repulsive_force_func(
            float* rep_func_outputs,
            int normalized,
            int frob,
            float dist,
            float a,
            float b,
            float cell_size,
            float average_weight
    )

ctypedef np.float32_t DTYPE_FLOAT
ctypedef np.int32_t DTYPE_INT

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void collect_attr_grads(
    float *attr_grads,
    int[:] head,
    int[:] tail,
    float[:, :] head_embedding,
    float[:, :] tail_embedding,
    float[:] weights,
    int normalized,
    int amplify_grads,
    int sym_attraction,
    int frob,
    int num_threads,
    int n_vertices,
    int n_edges,
    int i_epoch,
    int dim,
    float a,
    float b,
) nogil:
    cdef:
        int edge, j, k, d, v
        float dist, grad_d
        float *y1
        float *y2
        float attr_force

    for v in range(n_vertices):
        for d in range(dim):
            attr_grads[v * dim + d] = 0

    y1 = <float*> malloc(sizeof(float) * dim)
    y2 = <float*> malloc(sizeof(float) * dim)
    for edge in range(n_edges):
        j = head[edge]
        k = tail[edge]
        for d in range(dim):
            y1[d] = head_embedding[j, d]
            y2[d] = tail_embedding[k, d]
        dist = sq_euc_dist(y1, y2, dim)

        attr_force = attractive_force_func(
            normalized,
            frob,
            dist,
            a,
            b,
            weights[edge]
        )

        for d in range(dim):
            grad_d = attr_force * (y1[d] - y2[d])
            attr_grads[j * dim + d] -= grad_d
            if sym_attraction:
                attr_grads[k * dim + d] += grad_d
    free(y1)
    free(y2)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float collect_single_rep_grads(
    float *single_rep_grads,
    float[:, :] head_embedding,
    float[:, :] tail_embedding,
    int[:] head,
    int[:] tail,
    int normalized,
    int frob,
    int num_threads,
    int n_vertices,
    int n_edges,
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
        float dist, cell_dist, cell_size, grad_d
        float *rep_func_outputs
        float *y1
        float *y2

    for v in range(n_vertices):
        for d in range(dim):
            single_rep_grads[v * dim + d] = 0

    cdef float Z = 0
    rep_func_outputs = <float*> malloc(sizeof(float) * 2)
    y1 = <float*> malloc(sizeof(float) * dim)
    y2 = <float*> malloc(sizeof(float) * dim)
    for edge in range(n_edges):
        j = head[edge]
        k = rand() % n_vertices
        for d in range(dim):
            y1[d] = head_embedding[j, d]
            y2[d] = tail_embedding[k, d]

        dist = sq_euc_dist(y1, y2, dim)
        repulsive_force_func(
            rep_func_outputs,
            normalized,
            frob,
            dist,
            a,
            b,
            1,
            average_weight,
        )
        Z += rep_func_outputs[1]

        for d in range(dim):
            grad_d = rep_func_outputs[0]
            single_rep_grads[j * dim + d] += grad_d

    free(y1)
    free(y2)
    free(rep_func_outputs)

    return Z


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float collect_bh_rep_grads(
    float *bh_rep_grads,
    float[:, :] head_embedding,
    _QuadTree qt,
    int normalized,
    int frob,
    int num_threads,
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
        float dist, cell_dist, cell_size, grad_d
        float *cell_summaries
        float *rep_func_outputs
        float *y1

    for v in range(n_vertices):
        for d in range(dim):
            bh_rep_grads[v * dim + d] = 0

    cdef float Z = 0
    cell_summaries = <float*> malloc(sizeof(float) * n_vertices * offset)
    rep_func_outputs = <float*> malloc(sizeof(float) * 2)
    y1 = <float*> malloc(sizeof(float) * dim)
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
            repulsive_force_func(
                rep_func_outputs,
                normalized,
                frob,
                cell_dist,
                a,
                b,
                cell_size,
                average_weight,
            )
            Z += rep_func_outputs[1]

            for d in range(dim):
                dim_index = i_cell * offset + d
                grad_d = rep_func_outputs[0] * cell_summaries[dim_index]
                bh_rep_grads[v * dim + d] += grad_d

    free(cell_summaries)
    free(y1)
    free(rep_func_outputs)

    return Z

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _tsne_epoch(
    float[:] average_grad_dists,
    int normalized,
    int angular,
    int sym_attraction,
    int frob,
    int num_threads,
    int amplify_grads,
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
        float grad_scalar, grad_d, Z_bh, Z_single
        float theta = 0.5
        float ang_dist
        long dim_index, index
        float *bh_rep_grads
        float *single_rep_grads
        float *attr_grads

    cdef int n_edges = int(head.shape[0])
    cdef float attr_scalar = 4 * a * b
    cdef float rep_scalar_bh = 4 * a * b
    cdef float rep_scalar_single = 4 * a * b
    cdef float average_weight = get_avg_weight(&weights[0], n_edges)
    cdef float dist_sum = 0
    cdef float bh_norm = 0
    cdef float single_norm = 0

    # Allocte memory for data structures
    attr_grads = <float*> malloc(sizeof(float) * n_vertices * dim)
    bh_rep_grads = <float*> malloc(sizeof(float) * n_vertices * dim)
    single_rep_grads = <float*> malloc(sizeof(float) * n_vertices * dim)

    cdef float bh_grad_d
    cdef float single_grad_d

    cdef float result = 0.0
    cdef float bh_grad_len  = 0.0
    cdef float single_grad_len  = 0.0

    with nogil:
        collect_attr_grads(
            attr_grads,
            head,
            tail,
            head_embedding,
            tail_embedding,
            weights,
            normalized,
            amplify_grads,
            sym_attraction,
            frob,
            num_threads,
            n_vertices,
            n_edges,
            i_epoch,
            dim,
            a,
            b,
        )

        Z_bh = collect_bh_rep_grads(
            bh_rep_grads,
            head_embedding,
            qt,
            normalized,
            frob,
            num_threads,
            n_vertices,
            dim,
            a,
            b,
            average_weight
        )

        Z_single = collect_single_rep_grads(
            single_rep_grads,
            head_embedding,
            tail_embedding,
            head,
            tail,
            normalized,
            frob,
            num_threads,
            n_vertices,
            n_edges,
            dim,
            a,
            b,
            average_weight
        )

        if not normalized:
            Z_bh = 1
            Z_single = 1
            # Dividing by n_vertices means we have the same one-to-one
            #   relationship between attractive and repulsive forces
            #   as in traditional UMAP
            rep_scalar_bh /= n_vertices
            rep_scalar_single /= n_vertices

        for v in range(n_vertices):
            for d in range(dim):
                index = v * dim + d
                attr_grads[index] *= attr_scalar
                bh_rep_grads[index] *= rep_scalar_bh / Z_bh
                single_rep_grads[index] *= rep_scalar_single / Z_single

        for v in range(n_vertices):
            result = 0.0
            bh_grad_len = 0.0
            single_grad_len = 0.0
            for d in range(dim):
                bh_grad_d = bh_rep_grads[v * dim + d]
                single_grad_d = single_rep_grads[v * dim + d]

                result += bh_grad_d * single_grad_d
                bh_grad_len += bh_grad_d * bh_grad_d
                single_grad_len += single_grad_d * single_grad_d

            ang_dist = result / (sqrt(bh_grad_len) * sqrt(single_grad_len))
            dist_sum += ang_dist

        dist_sum /= n_vertices
        average_grad_dists[i_epoch] = dist_sum

        for v in range(n_vertices):
            for d in range(dim):
                index = v * dim + d
                if (bh_rep_grads[index] + attr_grads[index]) * all_updates[index] > 0.0:
                    gains[index] += 0.2
                else:
                    gains[index] *= 0.8
                gains[index] = clip(gains[index], 0.01, 1000)
                grad_d = (bh_rep_grads[index] + attr_grads[index]) * gains[index]

                if amplify_grads:
                    all_updates[index] = grad_d * lr + 0.9 * all_updates[index]
                else:
                    all_updates[index] = grad_d * lr

                head_embedding[v, d] += all_updates[index]

    free(attr_grads)
    free(bh_rep_grads)
    free(single_rep_grads)


cdef tsne_optimize(
    int normalized,
    int angular,
    int sym_attraction,
    int frob,
    int num_threads,
    int amplify_grads,
    float[:] average_grad_dists,
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
    """
    Wrapper to call barnes_hut optimization
    Require a regular def function to call from python file
    But this standard function in a .pyx file can call the cdef function
    """
    cdef:
        float *all_updates
        float *gains
    # Can only define cython quadtree in a cython function
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
        lr = get_lr(initial_lr, i_epoch, n_epochs, amplify_grads)
        _tsne_epoch(
            average_grad_dists,
            normalized,
            angular,
            sym_attraction,
            frob,
            num_threads,
            amplify_grads,
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

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def tsne_opt_wrapper(
    str optimize_method,
    int normalized,
    int angular,
    int sym_attraction,
    int frob,
    int num_threads,
    int amplify_grads,
    float[:, :] head_embedding,
    float[:, :] tail_embedding,
    int[:] head,
    int[:] tail,
    float[:] weights,
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
    average_grad_dists = py_np.zeros([n_epochs], dtype=py_np.float32)

    tsne_optimize(
        normalized,
        angular,
        sym_attraction,
        frob,
        num_threads,
        amplify_grads,
        average_grad_dists,
        head_embedding,
        tail_embedding,
        head,
        tail,
        weights,
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

    return py_np.asarray(head_embedding), average_grad_dists
