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
    float ang_dist(float* x, float* y, int dim)
cdef extern from "cython_utils.c" nogil:
    float get_lr(float initial_lr, int i_epoch, int n_epochs, int amplify_grads) 
cdef extern from "cython_utils.c" nogil:
    void print_status(int i_epoch, int n_epochs)
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
cdef void gather_gradients(
    float *local_Z,
    float *attr_grads,
    float *rep_grads,
    int[:] head,
    int[:] tail,
    float[:, :] head_embedding,
    float[:, :] tail_embedding,
    float[:] weights,
    int normalized,
    int angular,
    int amplify_grads,
    int frob,
    int sym_attraction,
    int num_threads,
    int n_vertices,
    int n_edges,
    int i_epoch,
    int dim,
    int negative_sample_rate,
    float a,
    float b,
    float average_weight
) nogil:
    cdef:
        int edge, j, k, d, s
        float dist, weight_scalar, grad_d
        float attr_force, rep_force
        float *y1
        float *y2
        float *rep_func_outputs

    with parallel(num_threads=num_threads):
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

            if angular:
                dist = ang_dist(y1, y2, dim)
            else:
                dist = sq_euc_dist(y1, y2, dim)

            # t-SNE early exaggeration
            if amplify_grads and i_epoch < 250:
                weight_scalar = 4
            else:
                weight_scalar = 1

            attr_force = attractive_force_func(
                normalized,
                frob,
                dist,
                a,
                b,
                weights[edge] * weight_scalar
            )
            for d in range(dim):
                grad_d = attr_force * (y1[d] - y2[d])
                attr_grads[j * dim + d] -= grad_d
                if sym_attraction:
                    k = tail[edge]
                    attr_grads[k * dim + d] += grad_d

            for s in range(negative_sample_rate):
                k = rand() % n_vertices
                for d in range(dim):
                    y2[d] = tail_embedding[k, d]

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
                    average_weight,
                )
                rep_force = rep_func_outputs[0]
                local_Z[0] += rep_func_outputs[1]

                for d in range(dim):
                    grad_d = rep_force * (y1[d] - y2[d])
                    rep_grads[j * dim + d] += grad_d

        free(rep_func_outputs)
        free(y1)
        free(y2)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _uniform_umap_epoch(
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
    float a,
    float b,
    int dim,
    int n_vertices,
    int negative_sample_rate,
    float lr,
    int i_epoch
):
    cdef:
        int i, j, k, d, v, index
        float grad_d
        float *attr_grads
        float *rep_grads
        float *all_grads
        float *local_Z

    cdef int n_edges = int(head.shape[0])
    cdef float Z = 0
    cdef float average_weight = get_avg_weight(&weights[0], n_edges)

    local_Z = <float*> malloc(sizeof(float))
    attr_grads = <float*> malloc(sizeof(float) * n_edges * dim)
    rep_grads = <float*> malloc(sizeof(float) * n_edges * dim)
    with nogil:
        for v in prange(n_vertices, num_threads=num_threads):
            for d in range(dim):
                attr_grads[v * dim + d] = 0
                rep_grads[v * dim + d] = 0

    with nogil:
        gather_gradients(
            local_Z,
            attr_grads,
            rep_grads,
            head,
            tail,
            head_embedding,
            tail_embedding,
            weights,
            normalized,
            angular,
            amplify_grads,
            frob,
            sym_attraction,
            num_threads,
            n_vertices,
            n_edges,
            i_epoch,
            dim,
            negative_sample_rate,
            a,
            b,
            average_weight
        )
        Z += local_Z[0]
        free(local_Z)
        if not normalized:
            Z = 1

    with nogil, parallel(num_threads=num_threads):
        for v in prange(n_vertices):
            for d in range(dim):
                index = v * dim + d
                grad_d = 4 * a * b * (attr_grads[index] + rep_grads[index] / Z)
                if grad_d * all_updates[index] > 0.0:
                    gains[index] += 0.2
                else:
                    gains[index] *= 0.8
                gains[index] = clip(gains[index], 0.01, 1000)
                grad_d = clip(grad_d * gains[index], -1, 1)

                all_updates[index] = grad_d * lr \
                                   + amplify_grads * 0.9 * all_updates[index]
                head_embedding[v, d] += all_updates[index]

    free(attr_grads)
    free(rep_grads)


cdef uniform_umap_optimize(
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
    float a,
    float b,
    int dim,
    float initial_lr,
    int n_epochs,
    int n_vertices,
    int negative_sample_rate,
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
        lr = get_lr(initial_lr, i_epoch, n_epochs, amplify_grads)
        _uniform_umap_epoch(
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
            a,
            b,
            dim,
            n_vertices,
            negative_sample_rate,
            lr,
            i_epoch
        )
        # FIXME -- why is this still printing even for verbose=False
        if verbose:
            print_status(i_epoch, n_epochs)

    free(all_updates)
    free(gains)



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def uniform_umap_opt_wrapper(
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
    float a,
    float b,
    float initial_lr,
    int negative_sample_rate,
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
        initial_lr = n_vertices / 500

    uniform_umap_optimize(
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
        a,
        b,
        dim,
        initial_lr,
        n_epochs,
        n_vertices,
        negative_sample_rate,
        verbose
    )

    return py_np.asarray(head_embedding)
