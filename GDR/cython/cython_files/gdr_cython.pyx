import numpy as py_np
cimport numpy as np
cimport cython
from libc.stdio cimport printf
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
from libc.math cimport sqrt
from libc.stdlib cimport rand
from libc.stdlib cimport malloc, free
from cython.parallel cimport prange, parallel
from sklearn.neighbors._quad_tree cimport _QuadTree
np.import_array()
INF = py_np.inf

cdef extern from "../utils/cython_utils.c" nogil:
    float clip(float value, float lower, float upper)
cdef extern from "../utils/cython_utils.c" nogil:
    float sq_euc_dist(float* x, float* y, int dim)
cdef extern from "../utils/cython_utils.c" nogil:
    float ang_dist(float* x, float* y, int dim)
cdef extern from "../utils/cython_utils.c" nogil:
    float get_lr(float initial_lr, int i_epoch, int n_epochs, int amplify_grads) 
cdef extern from "../utils/cython_utils.c" nogil:
    void print_status(int i_epoch, int n_epochs)
cdef extern from "../utils/cython_utils.c" nogil:
    float get_avg_weight(float* weights, int n_edges)
cdef extern from "../utils/cython_utils.c" nogil:
    float attractive_force_func(
            int normalized,
            int frob,
            float dist,
            float a,
            float b,
            float edge_weight
    )
cdef extern from "../utils/cython_utils.c" nogil:
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
cdef float gather_gradients(
    float *attr_grads,
    float *rep_grads,
    int[:] head,
    int[:] tail,
    float[:, :] head_embedding,
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
        float Z = 0

    # with parallel(num_threads=num_threads):
    y1 = <float*> malloc(sizeof(float) * dim)
    y2 = <float*> malloc(sizeof(float) * dim)
    rep_func_outputs = <float*> malloc(sizeof(float) * 2)
    for edge in prange(n_edges):
        j = head[edge]
        k = tail[edge]
        for d in range(dim):
            y1[d] = head_embedding[j, d]
            y2[d] = head_embedding[k, d]

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
                attr_grads[k * dim + d] += grad_d

        # FIXME -- should NOT be over neg_sample_rate
        for s in range(negative_sample_rate):
            k = rand() % n_vertices
            for d in range(dim):
                y2[d] = head_embedding[k, d]

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
            Z += rep_func_outputs[1]

            for d in range(dim):
                grad_d = rep_force * (y1[d] - y2[d])
                rep_grads[j * dim + d] += grad_d

    free(rep_func_outputs)
    free(y1)
    free(y2)

    return Z


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _gdr_epoch(
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
    float[:, :] all_updates,
    float[:, :] gains,
    float a,
    float b,
    int dim,
    int n_vertices,
    int negative_sample_rate,
    float lr,
    int i_epoch
):
    cdef:
        int i, j, k, v, index, s, edge
        int d, d1, d2, d3, d4
        # float *attr_grads
        # float *rep_grads
        float dist, weight_scalar, grad_d
        float attr_force, rep_force
        float *y1
        float *y2
        float *rep_func_outputs

    cdef int n_edges = int(head.shape[0])
    cdef float Z = 0
    cdef float average_weight = get_avg_weight(&weights[0], n_edges)

    py_attr_grads = py_np.zeros([n_edges, dim], dtype=py_np.float32)
    py_rep_grads = py_np.zeros([n_edges, dim], dtype=py_np.float32)
    py_Z_values = py_np.zeros([n_edges], dtype=py_np.float32)
    cdef float[:, :] attr_grads = py_attr_grads
    cdef float[:, :] rep_grads = py_rep_grads
    cdef float[:] Z_values = py_Z_values

    # Things that do NOT cause the slowdown on Freya/Odin
    #   - Collecting Z outside of gather_gradients function
    #   - parallel for loop processing of head_embedding updates
    #   - making the attr_grads and rep_grads thread-local inside gather_gradients
    #     and adding them up outside the threads

    cdef float *attr_grads_local
    cdef float *rep_grads_local
    cdef float *Z_values_local
    cdef double current
    with nogil, parallel(num_threads=num_threads):
        edge = 0
        # Z += gather_gradients(
        #     attr_grads,
        #     rep_grads,
        #     head,
        #     tail,
        #     head_embedding,
        #     weights,
        #     normalized,
        #     angular,
        #     amplify_grads,
        #     frob,
        #     sym_attraction,
        #     num_threads,
        #     n_vertices,
        #     n_edges,
        #     i_epoch,
        #     dim,
        #     negative_sample_rate,
        #     a,
        #     b,
        #     average_weight
        # )
        # if not normalized:
        #     Z = 1
        attr_grads_local = <float*> malloc(sizeof(float) * n_edges * dim)
        rep_grads_local = <float*> malloc(sizeof(float) * n_edges * dim)
        Z_values_local = <float*> malloc(sizeof(float) * n_edges)
        for v in range(n_vertices):
            Z_values_local[v] = 0
            for d in range(dim):
                attr_grads_local[v * dim + d] = 0
                rep_grads_local[v * dim + d] = 0

        y1 = <float*> malloc(sizeof(float) * dim)
        y2 = <float*> malloc(sizeof(float) * dim)
        rep_func_outputs = <float*> malloc(sizeof(float) * 2)
        for edge in range(n_edges):
            j = head[edge]
            k = tail[edge]
            for d1 in range(dim):
                y1[d1] = head_embedding[j, d1]
                y2[d1] = head_embedding[k, d1]

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
            for d2 in range(dim):
                grad_d = attr_force * (y1[d2] - y2[d2])
                attr_grads_local[j * dim + d2] -= grad_d
                if sym_attraction:
                    attr_grads_local[k * dim + d2] += grad_d

            # FIXME -- should NOT be over neg_sample_rate
            k = edge / n_vertices
            for d3 in range(dim):
                y2[d3] = head_embedding[k, d3]

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
            Z_values_local[edge] = rep_func_outputs[1]

            for d4 in range(dim):
                grad_d = rep_force * (y1[d4] - y2[d4])
                rep_grads_local[j * dim + d4] += grad_d

        with gil:
            edge = 0
            for edge in range(n_edges):
                Z_values[edge] = Z_values_local[edge]
            for v in range(n_vertices):
                for d in range(dim):
                    attr_grads[v, d] += attr_grads_local[v * dim + d]
                    rep_grads[v, d] += rep_grads_local[v * dim + d]

        free(rep_func_outputs)
        free(y1)
        free(y2)
        free(attr_grads_local)
        free(rep_grads_local)


    for edge in prange(n_edges, nogil=True, num_threads=num_threads):
        Z += Z_values[edge]

    for v in prange(n_vertices, nogil=True, num_threads=num_threads):
        for d in range(dim):
            index = v * dim + d
            grad_d = 4 * a * b * (attr_grads[v, d] + rep_grads[v, d] / Z)
            if grad_d * all_updates[v, d] > 0.0:
                gains[v, d] += 0.2
            else:
                gains[v, d] *= 0.8
            gains[v, d] = clip(gains[v, d], 0.01, 1000)
            grad_d = clip(grad_d * gains[v, d], -1, 1)

            all_updates[v, d] = grad_d * lr \
                               + amplify_grads * 0.9 * all_updates[v, d]
            head_embedding[v, d] += all_updates[v, d]


cdef gdr_optimize(
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
    int n_epochs,
    int n_vertices,
    int negative_sample_rate,
    int verbose
):
    cdef:
        int v, d, index

    py_all_updates = py_np.zeros([n_vertices, dim], dtype=py_np.float32)
    py_gains = py_np.ones([n_vertices, dim], dtype=py_np.float32)
    cdef float[:, :] all_updates = py_all_updates
    cdef float[:, :] gains = py_gains

    for i_epoch in range(n_epochs):
        lr = get_lr(initial_lr, i_epoch, n_epochs, amplify_grads)
        _gdr_epoch(
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



@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def gdr_opt_wrapper(
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

    gdr_optimize(
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
        n_epochs,
        n_vertices,
        negative_sample_rate,
        verbose
    )

    return py_np.asarray(head_embedding)
