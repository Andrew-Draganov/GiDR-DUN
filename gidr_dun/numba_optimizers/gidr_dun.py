import numpy as np
import numba
from tqdm.auto import tqdm

from .numba_utils import (
    clip,
    sq_euc_dist,
    ang_dist,
    get_lr,
    print_status,
    get_avg_weight,
    kernel_function,
    umap_rep_scalar,
    attractive_force_func,
    repulsive_force_func
)

@numba.njit(fastmath=True, parallel=True, boundscheck=False)
def gather_gradients(
    attr_grads,
    rep_grads,
    head,
    tail,
    head_embedding,
    tail_embedding,
    weights,
    weight_scalar,
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
    a,
    b,
    average_weight
):
    Z = 0
    for edge in numba.prange(n_edges):
        j = head[edge]
        k = tail[edge]
        y1 = head_embedding[j]
        y2 = tail_embedding[k]
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

        k = np.random.randint(0, n_vertices)
        y2 = tail_embedding[k]
        dist = sq_euc_dist(y1, y2, dim)

        if normalized or frob:
            q = kernel_function(dist, a, b)
        else:
            q = umap_rep_scalar(dist, a, b)

        rep_force = repulsive_force_func(normalized, frob, q, average_weight)
        Z += q

        for d in range(dim):
            grad_d = rep_force * (y1[d] - y2[d])
            rep_grads[j, d] += grad_d

    return Z


@numba.njit(fastmath=True, parallel=True, boundscheck=False)
def apply_forces(
    amplify_grads,
    head_embedding,
    attr_grads,
    rep_grads,
    gains,
    all_updates,
    Z,
    a,
    b,
    n_vertices,
    dim,
    lr
):
    for v in numba.prange(n_vertices):
        for d in range(dim):
            grad_d = 4 * a * b * (attr_grads[v, d] + rep_grads[v, d] / Z)
            if grad_d * all_updates[v, d] > 0.0:
                gains[v, d] += 0.2
            else:
                gains[v, d] *= 0.8
            gains[v, d] = clip(gains[v, d], 0.01, 1000)
            grad_d = clip(grad_d * gains[v, d], -1, 1)

            all_updates[v, d] = grad_d * lr + amplify_grads * 0.9 * all_updates[v, d]
            head_embedding[v, d] += all_updates[v, d]



def gidr_dun_numba_wrapper(
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
    n_epochs,
    n_vertices,
    a,
    b,
    initial_lr,
    negative_sample_rate,
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
        initial_lr = n_vertices / 500

    all_updates = np.zeros([n_vertices, dim], dtype=np.float32)
    gains = np.ones([n_vertices, dim], dtype=np.float32)

    n_edges = int(head.shape[0])
    attr_grads = np.zeros([n_edges, dim], dtype=np.float32)
    rep_grads = np.zeros([n_edges, dim], dtype=np.float32)
    average_weight = get_avg_weight(weights, n_edges)

    for i_epoch in tqdm(range(n_epochs)):
        lr = get_lr(initial_lr, i_epoch, n_epochs, amplify_grads)
        for v in range(n_vertices):
            for d in range(dim):
                attr_grads[v, d] = 0
                rep_grads[v, d] = 0

        # t-SNE early exaggeration
        if amplify_grads and i_epoch < 250:
            weight_scalar = 4
        else:
            weight_scalar = 1

        Z = gather_gradients(
            attr_grads,
            rep_grads,
            head,
            tail,
            head_embedding,
            tail_embedding,
            weights,
            weight_scalar,
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
            a,
            b,
            average_weight
        )
        if not normalized:
            Z = 1

        apply_forces(
            amplify_grads,
            head_embedding,
            attr_grads,
            rep_grads,
            gains,
            all_updates,
            Z,
            a,
            b,
            n_vertices,
            dim,
            lr
        )

    return head_embedding
