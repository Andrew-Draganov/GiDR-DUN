import numpy as np
import numba
from GDR.optimizer.utils import tau_rand_int
from tqdm.auto import tqdm

from .numba_utils import (
    clip,
    sq_euc_dist,
    ang_dist,
    get_lr,
    print_status,
    get_avg_weight,
    kernel_function,
    umap_attr_scalar,
    umap_rep_scalar,
    attractive_force_func,
    repulsive_force_func
)

def gdr_single_epoch(
    normalized,
    standard_opt,
    angular,
    sym_attraction,
    frob,
    num_threads,
    amplify_grads,
    embedding,
    head,
    tail,
    q_ij,
    q_ik,
    rand_indices,
    attr_grads,
    rep_grads,
    all_updates,
    gains,
    weights,
    n_epochs,
    n_vertices,
    a,
    b,
    rng_state,
    initial_lr,
    negative_sample_rate,
    n_edges,
    dim,
    i_epoch,
    average_weight,
    lr,
    epochs_per_sample,
    epoch_of_next_sample
):
    for v in numba.prange(n_vertices):
        for d in range(dim):
            attr_grads[v, d] = 0.0
            rep_grads[v, d] = 0.0

    Z = 0.0
    for edge in numba.prange(n_edges):
        if epoch_of_next_sample[edge] <= i_epoch or standard_opt:
            j = head[edge]
            k = tail[edge]
            y1 = embedding[j]
            y2 = embedding[k]
            dist = sq_euc_dist(y1, y2, dim)
            q_ij[edge] = kernel_function(dist, a, b)
            # Z += q_ik[edge] # FIXME

            k = tau_rand_int(rng_state) % n_vertices
            y2 = embedding[k]
            dist = sq_euc_dist(y1, y2, dim)
            q_ik[edge] = kernel_function(dist, a, b)
            rand_indices[edge] = k
            Z += q_ik[edge]

    sum_pq = 0.0
    sum_qq = 0.0
    for edge in numba.prange(n_edges):
        if epoch_of_next_sample[edge] <= i_epoch or standard_opt:
            q_ij[edge] /= Z
            q_ik[edge] /= Z
            sum_pq += weights[edge] * q_ij[edge]
            sum_qq += q_ik[edge] * q_ik[edge]

    for edge in numba.prange(n_edges):
        j = head[edge]
        k = tail[edge]
        y1 = embedding[j]
        y2 = embedding[k]
        q_sq = q_ij[edge] * q_ij[edge]
        attr_force = (weights[edge] - sum_pq) * q_sq

        for d in range(dim):
            grad_d = attr_force * (y1[d] - y2[d])
            attr_grads[j, d] -= grad_d
            if sym_attraction:
                attr_grads[k, d] += grad_d

        y2 = embedding[rand_indices[edge]]
        q_sq = q_ik[edge] * q_ik[edge]
        rep_force = (sum_qq - q_ik[edge]) * q_sq

        for d in range(dim):
            grad_d = rep_force * (y1[d] - y2[d])
            rep_grads[j, d] += grad_d

    for v in numba.prange(n_vertices):
        for d in range(dim):
            grad_d = 4.0 * a * b * (attr_grads[v, d] + rep_grads[v, d]) * Z
            grad_d = clip(grad_d, -1.0, 1.0)
            embedding[v, d] += grad_d * lr

def gdr_numba_wrapper(
    normalized,
    standard_opt,
    angular,
    sym_attraction,
    frob,
    num_threads,
    amplify_grads,
    head_embedding,
    head,
    tail,
    weights,
    n_epochs,
    n_vertices,
    a,
    b,
    rng_state,
    initial_lr,
    negative_sample_rate,
    epochs_per_sample,
    verbose=True,
    **kwargs
):
    dim = int(head_embedding.shape[1])
    # Perform weight scaling on high-dimensional relationships
    if normalized:
        weight_sum = 0.0
        for i in range(weights.shape[0]):
            weight_sum += weights[i]
        for i in range(weights.shape[0]):
            weights[i] /= weight_sum

    all_updates = np.zeros([n_vertices, dim], dtype=np.float32)
    gains = np.ones([n_vertices, dim], dtype=np.float32)

    n_edges = int(head.shape[0])
    attr_grads = np.zeros([n_edges, dim], dtype=np.float32)
    rep_grads = np.zeros([n_edges, dim], dtype=np.float32)
    average_weight = get_avg_weight(weights, n_edges)
    epoch_of_next_sample = epochs_per_sample.copy()

    optimize_fn = numba.njit(
        gdr_single_epoch,
        fastmath=True,
        parallel=True,
        boundscheck=False
    )

    for i_epoch in tqdm(range(1, n_epochs+1)):
        lr = get_lr(initial_lr, i_epoch, n_epochs, amplify_grads)
        q_ij = np.zeros([n_edges], dtype=np.float32)
        q_ik = np.zeros([n_edges], dtype=np.float32)
        rand_indices = np.zeros([n_edges], dtype=np.int32)
        optimize_fn(
            normalized,
            standard_opt,
            angular,
            sym_attraction,
            frob,
            num_threads,
            amplify_grads,
            head_embedding,
            head,
            tail,
            q_ij,
            q_ik,
            rand_indices,
            attr_grads,
            rep_grads,
            all_updates,
            gains,
            weights,
            n_epochs,
            n_vertices,
            a,
            b,
            rng_state,
            initial_lr,
            negative_sample_rate,
            n_edges,
            dim,
            i_epoch,
            average_weight,
            lr,
            epochs_per_sample,
            epoch_of_next_sample
        )

    return head_embedding
