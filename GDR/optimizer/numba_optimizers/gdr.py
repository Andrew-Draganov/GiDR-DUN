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
    # t-SNE early exaggeration
    if amplify_grads and i_epoch < 250:
        weight_scalar = 4.0
    else:
        weight_scalar = 1.0

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
                attr_grads[j, d] -= grad_d
                if sym_attraction:
                    attr_grads[k, d] += grad_d

            k = tau_rand_int(rng_state) % n_vertices
            y2 = embedding[k]
            dist = sq_euc_dist(y1, y2, dim)

            rep_force, q = repulsive_force_func(
                normalized,
                frob,
                dist,
                a,
                b,
                average_weight
            )
            Z += q

            for d in range(dim):
                grad_d = rep_force * (y1[d] - y2[d])
                rep_grads[j, d] += grad_d

            epoch_of_next_sample[edge] += epochs_per_sample[edge]

    if not normalized:
        Z = 1.0

    for v in numba.prange(n_vertices):
        for d in range(dim):
            grad_d = 4.0 * a * b * (attr_grads[v, d] + rep_grads[v, d] / Z)

            if grad_d * all_updates[v, d] > 0.0:
                gains[v, d] += 0.2
            else:
                gains[v, d] *= 0.8
            gains[v, d] = clip(gains[v, d], 0.01, 1000)
            grad_d = clip(grad_d * gains[v, d], -1.0, 1.0)

            all_updates[v, d] = grad_d * lr + amplify_grads * 0.9 * all_updates[v, d]
            embedding[v, d] += all_updates[v, d]

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
