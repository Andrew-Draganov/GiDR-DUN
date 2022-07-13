import numpy as py_np
cimport numpy as np
cimport cython
from libc.stdio cimport printf
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange, parallel
from sklearn.neighbors._quad_tree cimport _QuadTree
np.import_array()

INF = py_np.inf

cdef extern from "../utils/cython_utils.cpp" nogil:
    float clip(float value, float lower, float upper)
cdef extern from "../utils/cython_utils.cpp" nogil:
    float sq_euc_dist(float* x, float* y, int dim)
cdef extern from "../utils/cython_utils.cpp" nogil:
    float ang_dist(float* x, float* y, int dim)
cdef extern from "../utils/cython_utils.cpp" nogil:
    float get_lr(float initial_lr, int i_epoch, int n_epochs, int amplify_grads) 
cdef extern from "../utils/cython_utils.cpp" nogil:
    void print_status(int i_epoch, int n_epochs)
cdef extern from "../utils/cython_utils.cpp" nogil:
    float umap_repulsion_grad(float dist, float a, float b)
cdef extern from "../utils/cython_utils.cpp" nogil:
    float kernel_function(float dist, float a, float b)
cdef extern from "../utils/cython_utils.cpp" nogil:
    float get_avg_weight(float* weights, int n_edges)
cdef extern from "../utils/cython_utils.cpp" nogil:
    float attractive_force_func(
            int normalized,
            int frob,
            float dist,
            float a,
            float b,
            float edge_weight
    )
cdef extern from "../utils/cython_utils.cpp" nogil:
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _tsne_epoch(
    int normalized,
    int angular,
    int sym_attraction,
    int frob,
    int num_threads,
    int amplify_grads,
    float[:, :] head_embedding,
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
        float grad_scalar, dist, grad_d, Z
        float theta = 0.5
        long index
        float *attr_grads
        float *rep_grads
        int edge
        float weight_scalar
        float *y1
        float *y2
        float attr_force
        float rep_force
        int dim_index
        long cell_metadata
        long offset = dim + 2
        float cell_dist, cell_size
        float *cell_summaries
        float *rep_func_outputs

    cdef int n_edges = int(head.shape[0])
    cdef float attr_scalar = 4 * a * b
    cdef float rep_scalar = 4 * a * b
    cdef float average_weight = get_avg_weight(&weights[0], n_edges)

    Z = 0

    # Allocte memory for data structures
    attr_grads = <float*> malloc(sizeof(float) * n_vertices * dim)
    rep_grads = <float*> malloc(sizeof(float) * n_vertices * dim)
    for v in range(n_vertices):
        for d in range(dim):
            index = v * dim + d
            attr_grads[index] = 0
            rep_grads[index] = 0

    # t-SNE early exaggeration
    if amplify_grads and i_epoch < 250:
        weight_scalar = 4
    else:
        weight_scalar = 1

    # Do the meat of the calculations in parallel
    with nogil, parallel(num_threads=num_threads):
        ### Get one attractive force for each edge ###
        # Zero out edge-wise forces
        for v in prange(n_vertices):
            for d in range(dim):
                attr_grads[v * dim + d] = 0

        y1 = <float*> malloc(sizeof(float) * dim)
        y2 = <float*> malloc(sizeof(float) * dim)
        for edge in prange(n_edges):
            j = head[edge]
            k = tail[edge]
            for d in range(dim):
                y1[d] = head_embedding[j, d]
                y2[d] = head_embedding[k, d]
            dist = sq_euc_dist(y1, y2, dim)

            # Calculate attractive force
            attr_force = attractive_force_func(
                normalized,
                frob,
                dist,
                a,
                b,
                weights[edge] * weight_scalar
            )

            # Add this force to the set acting on this point
            for d in range(dim):
                grad_d = attr_force * (y1[d] - y2[d])
                attr_grads[j * dim + d] -= grad_d
                if sym_attraction:
                    attr_grads[k * dim + d] += grad_d
        free(y2)


        # Get one repulsive force for each (edge, quadtree-cell) pair

        # Zero out edge-wise forces
        for v in prange(n_vertices):
            for d in range(dim):
                rep_grads[v * dim + d] = 0

        cell_summaries = <float*> malloc(sizeof(float) * n_vertices * offset)
        rep_func_outputs = <float*> malloc(sizeof(float) * 2)
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
                # Calculate repulsive force
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
                rep_force = rep_func_outputs[0]
                Z += rep_func_outputs[1]

                # Add this force to the set acting on this point
                for d in range(dim):
                    dim_index = i_cell * offset + d
                    grad_d = rep_func_outputs[0] * cell_summaries[dim_index]
                    rep_grads[v * dim + d] += grad_d

        free(cell_summaries)
        free(y1)
        free(rep_func_outputs)

    if not normalized:
        Z = 1
        # Dividing by n_vertices means we have the same one-to-one
        #   relationship between attractive and repulsive forces
        #   as in traditional UMAP
        rep_scalar /= n_vertices
    rep_scalar /= Z

    # Clean up and rescale all forces
    with nogil, parallel(num_threads=num_threads):
        for v in prange(n_vertices):
            for d in range(dim):
                index = v * dim + d
                attr_grads[index] *= attr_scalar
                rep_grads[index] *= rep_scalar

    # Perform gradient descent
    with nogil, parallel(num_threads=num_threads):
        for v in prange(n_vertices):
            for d in range(dim):
                index = v * dim + d
                if (rep_grads[index] + attr_grads[index]) * all_updates[index] > 0.0:
                    gains[index] += 0.2
                else:
                    gains[index] *= 0.8
                gains[index] = clip(gains[index], 0.01, 1000)
                grad_d = (rep_grads[index] + attr_grads[index]) * gains[index]

                all_updates[index] = grad_d * lr + amplify_grads * 0.9 * all_updates[index]
                head_embedding[v, d] += clip(all_updates[index], -1, 1)

    free(attr_grads)
    free(rep_grads)

cdef tsne_optimize(
    int normalized,
    int angular,
    int sym_attraction,
    int frob,
    int num_threads,
    int amplify_grads,
    float[:, :] head_embedding,
    int[:] head,
    int[:] tail,
    float[:] weights,
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
            normalized,
            angular,
            sym_attraction,
            frob,
            num_threads,
            amplify_grads,
            head_embedding,
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
    int[:] head,
    int[:] tail,
    float[:] weights,
    int n_epochs,
    int n_vertices,
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

    tsne_optimize(
        normalized,
        angular,
        sym_attraction,
        frob,
        num_threads,
        amplify_grads,
        head_embedding,
        head,
        tail,
        weights,
        a,
        b,
        dim,
        initial_lr,
        negative_sample_rate,
        n_epochs,
        n_vertices,
        verbose
    )

    return py_np.asarray(head_embedding)
