# distutils: language=c++
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

INF = py_np.inf

cdef extern from "../utils/cython_utils.cpp" nogil:
    float clip(float value, float lower, float upper)
cdef extern from "../utils/cython_utils.cpp" nogil:
    int random_int(const int & max)
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
cdef void _umap_epoch(
    int normalized,
    int angular,
    int sym_attraction,
    int frob,
    int num_threads,
    int amplify_grads,
    float[:, :] head_embedding,
    int[:] head,
    int[:] tail,
    float[:, :] attr_grads,
    float[:, :] rep_grads,
    float[:] weights,
    float[:, :] all_updates,
    float a,
    float b,
    int dim,
    int n_vertices,
    float lr,
    float[:] epochs_per_sample,
    float[:] epochs_per_negative_sample,
    float[:] epoch_of_next_negative_sample,
    float[:] epoch_of_next_sample,
    int i_epoch,
):
    cdef:
        int i, j, k, n_neg_samples, edge, p
        int index
        # Can't reuse loop variables in a `with nogil, parallel():` block
        int v1, v2
        int d1, d2, d3, d4, d5, d6
        float grad_d1, grad_d2, grad_d3, grad_d4
        float weight_scalar = 1
        float attr_force, rep_force
        float dist

    # FIXME FIXME -- normalized doesn't work here!
    cdef float Z = 0
    cdef int n_edges = int(head.shape[0])

    if amplify_grads and i_epoch < 250:
        weight_scalar = 4
    else:
        weight_scalar = 1

    cdef float *y1
    cdef float *y2
    cdef float *rep_func_outputs
    with nogil, parallel(num_threads=num_threads):
        # Zero out forces between epochs
        for v1 in prange(n_vertices):
            for d1 in range(dim):
                attr_grads[v1, d1] = 0
                rep_grads[v1, d1] = 0

        rep_func_outputs = <float*> malloc(sizeof(float) * 2)
        y1 = <float*> malloc(sizeof(float) * dim)
        y2 = <float*> malloc(sizeof(float) * dim)
        for edge in prange(n_edges):
            if epoch_of_next_sample[edge] <= i_epoch:
                # Gets one of the knn in HIGH-DIMENSIONAL SPACE relative to the sample point
                j = head[edge]
                k = tail[edge]

                for d2 in range(dim):
                    y1[d2] = head_embedding[j, d2]
                    y2[d2] = head_embedding[k, d2]

                if angular:
                    dist = ang_dist(y1, y2, dim)
                else:
                    dist = sq_euc_dist(y1, y2, dim)

                attr_force = attractive_force_func(
                    normalized,
                    frob,
                    dist,
                    a,
                    b,
                    1.0 * weight_scalar
                )

                for d3 in range(dim):
                    grad_d1 = clip(attr_force * (y1[d3] - y2[d3]), -4, 4)
                    attr_grads[j, d3] -= grad_d1
                    if sym_attraction:
                        attr_grads[k, d3] += grad_d1

                epoch_of_next_sample[edge] += epochs_per_sample[edge]
                n_neg_samples = int(
                    (i_epoch - epoch_of_next_negative_sample[edge]) / epochs_per_negative_sample[edge]
                )

                for p in range(n_neg_samples):
                    k = random_int(n_vertices)
                    for d4 in range(dim):
                        y2[d4] = head_embedding[k, d4]

                    if angular:
                        dist = ang_dist(y1, y2, dim)
                    else:
                        dist = sq_euc_dist(y1, y2, dim)

                    repulsive_force_func(
                        rep_func_outputs,
                        normalized,
                        frob,
                        dist,
                        a,
                        b,
                        1.0,
                        0.0
                    )
                    rep_force = rep_func_outputs[0]
                    Z += rep_func_outputs[1]

                    for d5 in range(dim):
                        grad_d2 = clip(rep_force * (y1[d5] - y2[d5]), -4, 4)
                        rep_grads[j, d5] += grad_d2

                epoch_of_next_negative_sample[edge] += (
                    n_neg_samples * epochs_per_negative_sample[edge]
                )

        free(y1)
        free(y2)
        free(rep_func_outputs)

    if not normalized or Z == 0:
        Z = 1

    with nogil, parallel(num_threads=num_threads):
        for v2 in prange(n_vertices):
            for d6 in range(dim):
                grad_d3 = rep_grads[v2, d6] / Z + attr_grads[v2, d6]
                all_updates[v2, d6] = grad_d3 * lr + amplify_grads * 0.9 * all_updates[v2, d6]
                head_embedding[v2, d6] += all_updates[v2, d6]


cdef umap_optimize(
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
        int i, i_epoch

    py_all_updates = py_np.zeros([n_vertices, dim], dtype=py_np.float32)
    cdef float[:, :] all_updates = py_all_updates

    # Get sample frequency arrays. Determines how often each edge is optimized along
    # These names are legacy umap names so they can more easily be found in the
    #   original UMAP repository
    py_epochs_per_negative_sample = py_np.zeros_like(epochs_per_sample)
    py_epoch_of_next_negative_sample = py_np.zeros_like(epochs_per_sample)
    py_epoch_of_next_sample = py_np.zeros_like(epochs_per_sample)
    cdef float[:] epochs_per_negative_sample = py_np.zeros_like(epochs_per_sample)
    cdef float[:] epoch_of_next_negative_sample = py_np.zeros_like(epochs_per_sample)
    cdef float[:] epoch_of_next_sample = py_np.zeros_like(epochs_per_sample)
    # Perform negative samples x times more often
    #   by making the number of epochs between samples smaller
    for i in range(weights.shape[0]):
        epochs_per_negative_sample[i] = epochs_per_sample[i] / negative_sample_rate
        epoch_of_next_negative_sample[i] = epochs_per_negative_sample[i]
        epoch_of_next_sample[i] = epochs_per_sample[i]

    cdef int n_edges = int(head.shape[0])
    py_attr_grads = py_np.zeros([n_edges, dim], dtype=py_np.float32)
    py_rep_grads = py_np.zeros([n_edges, dim], dtype=py_np.float32)
    cdef float[:, :] attr_grads = py_attr_grads
    cdef float[:, :] rep_grads = py_rep_grads

    for v in range(n_vertices):
        for d in range(dim):
            all_updates[v, d] = 0

    for i_epoch in range(n_epochs):
        lr = get_lr(initial_lr, i_epoch, n_epochs, amplify_grads)
        _umap_epoch(
            normalized,
            angular,
            sym_attraction,
            frob,
            num_threads,
            amplify_grads,
            head_embedding,
            head,
            tail,
            attr_grads,
            rep_grads,
            weights,
            all_updates,
            a,
            b,
            dim,
            n_vertices,
            lr,
            epochs_per_sample,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            i_epoch,
        )
        if verbose:
            print_status(i_epoch, n_epochs)


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def umap_opt_wrapper(
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
    float[:] epochs_per_sample,
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

    umap_optimize(
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

    return py_np.asarray(head_embedding)
