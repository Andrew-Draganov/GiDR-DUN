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

cdef extern from "cython_utils.c" nogil:
    float clip(float value, float lower, float upper)
cdef extern from "cython_utils.c" nogil:
    float sq_euc_dist(float* x, float* y, int dim)
cdef extern from "cython_utils.c" nogil:
    float get_lr(float initial_lr, int i_epoch, int n_epochs) 
cdef extern from "cython_utils.c" nogil:
    void print_status(int i_epoch, int n_epochs)
cdef extern from "cython_utils.c" nogil:
    float umap_repulsion_grad(float dist_squared, float a, float b)
cdef extern from "cython_utils.c" nogil:
    float kernel_function(float dist_squared, float a, float b)
cdef extern from "cython_utils.c" nogil:
    float attractive_force_func(
            int normalized,
            int frob,
            float dist_squared,
            float a,
            float b,
            float edge_weight
    )
cdef extern from "cython_utils.c" nogil:
    void repulsive_force_func(
            float* rep_func_outputs,
            int normalized,
            int frob,
            float dist_squared,
            float a,
            float b,
            float cell_size,
            float average_weight
    )

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
@cython.wraparound(False)
@cython.cdivision(True)
cdef void get_kernels(
    float *local_Z,
    float *attr_forces,
    float *rep_forces,
    float *attr_vecs,
    float *rep_vecs,
    int[:] head,
    int[:] tail,
    float[:, :] head_embedding,
    float[:, :] tail_embedding,
    float[:] weights,
    int normalized,
    int frob,
    int n_vertices,
    int n_edges,
    int i_epoch,
    int dim,
    float a,
    float b,
    float average_weight
) nogil:
    cdef:
        int edge, j, k, d
        float dist_squared, weight_scalar
        float *y1
        float *y2
        float *rep_func_outputs

    with parallel():
        local_Z[0] = 0
        y1 = <float*> malloc(sizeof(float) * dim)
        y2 = <float*> malloc(sizeof(float) * dim)
        rep_func_outputs = <float*> malloc(sizeof(float) * 2)
        for edge in prange(n_edges):
            j = head[edge]
            k = tail[edge]
            for d in range(dim):
                y1[d] = head_embedding[j, d]
                y2[d] = tail_embedding[k, d]
                attr_vecs[edge * dim + d] = y1[d] - y2[d]
            dist_squared = sq_euc_dist(y1, y2, dim)

            # t-SNE early exaggeration
            if i_epoch < 100:
                weight_scalar = 4
            else:
                weight_scalar = 1

            attr_forces[edge] = attractive_force_func(
                normalized,
                frob,
                dist_squared,
                a,
                b,
                weights[edge] * weight_scalar
            )

            k = rand() % n_vertices
            for d in range(dim):
                y2[d] = tail_embedding[k, d]
                rep_vecs[edge * dim + d] = y1[d] - y2[d]
            dist_squared = sq_euc_dist(y1, y2, dim)
            repulsive_force_func(
                rep_func_outputs,
                normalized,
                frob,
                dist_squared,
                a,
                b,
                1.0,
                average_weight,
            )
            rep_forces[edge] = rep_func_outputs[0]
            local_Z[0] += rep_func_outputs[1]

        free(rep_func_outputs)
        free(y1)
        free(y2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void gather_gradients(
    float *local_grads,
    int[:] head,
    int[:] tail,
    float* attr_forces,
    float* rep_forces,
    float* attr_vecs,
    float* rep_vecs,
    int sym_attraction,
    int frob,
    int n_vertices,
    int n_edges,
    int dim,
    float Z
) nogil:
    cdef:
        int j, k, v, d, edge, index
        float force, grad_d

    with parallel():
        # Fill allocated memory with zeros
        for v in prange(n_vertices):
            for d in range(dim):
                index = v * dim + d
                local_grads[index] = 0

        for edge in prange(n_edges):
            j = head[edge]
            for d in range(dim):
                grad_d = clip(attr_forces[edge] * attr_vecs[edge * dim + d], -1, 1)
                local_grads[j * dim + d] -= grad_d
                if sym_attraction:
                    k = tail[edge]
                    local_grads[k * dim + d] += grad_d

            for d in range(dim):
                grad_d = clip(rep_forces[edge] * rep_vecs[edge * dim + d], -1, 1)
                local_grads[j * dim + d] += grad_d / Z


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _uniform_umap_epoch(
    int normalized,
    int sym_attraction,
    int frob,
    int momentum,
    float[:, :] head_embedding,
    float[:, :] tail_embedding,
    int[:] head,
    int[:] tail,
    float[:] weights,
    float* all_updates,
    float* gains,
    float a,
    float b,
    int dim,
    int n_vertices,
    float lr,
    int i_epoch
):
    cdef:
        int i, j, k, d, v, index, edge
        float *all_grads
        float *local_grads

        float *attr_vecs
        float *rep_vecs
        float *attr_forces
        float *rep_forces

        float Z = 0
        float *local_Z
        float scalar

    cdef int n_edges = int(head.shape[0])
    local_Z = <float*> malloc(sizeof(float))
    attr_forces = <float*> malloc(sizeof(float) * n_edges)
    rep_forces = <float*> malloc(sizeof(float) * n_edges)
    attr_vecs = <float*> malloc(sizeof(float) * n_edges * dim)
    rep_vecs = <float*> malloc(sizeof(float) * n_edges * dim)

    all_grads = <float*> malloc(sizeof(float) * n_vertices * dim)
    local_grads = <float*> malloc(sizeof(float) * n_vertices * dim)
    cdef float grad_d = 0.0

    cdef float average_weight = get_avg_weight(weights)

    with nogil:
        for v in prange(n_vertices):
            for d in range(dim):
                all_grads[v * dim + d] = 0

    with nogil:
        get_kernels(
            local_Z,
            attr_forces,
            rep_forces,
            attr_vecs,
            rep_vecs,
            head,
            tail,
            head_embedding,
            tail_embedding,
            weights,
            normalized,
            frob,
            n_vertices,
            n_edges,
            i_epoch,
            dim,
            a,
            b,
            average_weight
        )

        Z += local_Z[0]
        free(local_Z)
        if not normalized and not frob:
            Z = 1

        gather_gradients(
            local_grads,
            head,
            tail,
            attr_forces,
            rep_forces,
            attr_vecs,
            rep_vecs,
            sym_attraction,
            frob,
            n_vertices,
            n_edges,
            dim,
            Z
        )
        free(attr_forces)
        free(rep_forces)
        free(attr_vecs)
        free(rep_vecs)

        # Need to collect thread-local forces into single gradient
        #   before performing gradient descent
        scalar = 4 * a * b
        for v in range(n_vertices):
            for d in range(dim):
                index = v * dim + d
                all_grads[index] += local_grads[index] * scalar
        free(local_grads)

        for v in prange(n_vertices):
            for d in range(dim):
                index = v * dim + d

                if all_grads[index] * all_updates[index] > 0.0:
                    gains[index] += 0.2
                else:
                    gains[index] *= 0.8
                gains[index] = clip(gains[index], 0.01, 100)
                grad_d = all_grads[index] * gains[index]

                all_updates[index] = grad_d * lr + momentum * 0.9 * all_updates[index]
                head_embedding[v, d] += all_updates[index]

    free(all_grads)


cdef uniform_umap_optimize(
    int normalized,
    int sym_attraction,
    int frob,
    int momentum,
    float[:, :] head_embedding,
    float[:, :] tail_embedding,
    int[:] head,
    int[:] tail,
    float[:] weights,
    float a,
    float b,
    int dim,
    float initial_lr,
    int n_epochs,
    int n_vertices,
    int verbose
):
    cdef:
        int v, d, index
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
        _uniform_umap_epoch(
            normalized,
            sym_attraction,
            frob,
            momentum,
            head_embedding,
            tail_embedding,
            head,
            tail,
            weights,
            all_updates,
            gains,
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
def uniform_umap_opt_wrapper(
    str optimize_method,
    int normalized,
    int sym_attraction,
    int frob,
    int momentum,
    float[:, :] head_embedding,
    float[:, :] tail_embedding,
    int[:] head,
    int[:] tail,
    float[:] weights,
    int n_epochs,
    int n_vertices,
    float a,
    float b,
    float initial_lr,
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

    uniform_umap_optimize(
        normalized,
        sym_attraction,
        frob,
        momentum,
        head_embedding,
        tail_embedding,
        head,
        tail,
        weights,
        a,
        b,
        dim,
        initial_lr,
        n_epochs,
        n_vertices,
        verbose
    )

    return py_np.asarray(head_embedding)
