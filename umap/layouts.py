import time
import numpy as np
import numba
import distances as dist
from sklearn.neighbors._quad_tree import _QuadTree
from utils import tau_rand_int
import pyximport
pyximport.install(setup_args={"script_args" : ["--verbose"]})
import comparison_utils


@numba.njit()
def clip(val):
    """Standard clamping of a value into a fixed range (in this case -4.0 to
    4.0)

    Parameters
    ----------
    val: float
        The value to be clamped.

    Returns
    -------
    The clamped value, now fixed to be in the range -4.0 to 4.0.
    """
    if val > 4.0:
        return 4.0
    elif val < -4.0:
        return -4.0
    else:
        return val

@numba.njit(
    "f4(f4[::1],f4[::1])",
    fastmath=True,
    cache=True,
    locals={
        "result": numba.types.float32,
        "diff": numba.types.float32,
        "dim": numba.types.intp,
    },
)
def rdist(x, y):
    """Reduced Euclidean distance.

    Parameters
    ----------
    x: array of shape (embedding_dim,)
    y: array of shape (embedding_dim,)

    Returns
    -------
    The squared euclidean distance between x and y
    """
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result


def optimize_through_sampling(
    normalized,
    sym_attraction,
    head_embedding,
    tail_embedding,
    head,
    tail,
    weights,
    grads,
    n_vertices,
    average_weight,
    epochs_per_sample,
    a,
    b,
    rng_state,
    dim,
    lr,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    i_epoch,
):
    """
    This is the original UMAP optimization function
    """
    # ANDREW - If we optimize an edge, then the next epoch we optimize it is
    #          the current epoch + epochs_per_sample[i] for that edge
    #        - Basically, optimize edge i every epochs_per_sample[i] steps
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= i_epoch:
            # Gets the knn in HIGH-DIMENSIONAL SPACE relative to the sample point
            j = head[i]
            k = tail[i]

            # ANDREW - pick random vertex from knn for calculating attractive force
            # t-SNE sums over all knn's attractive forces
            current = head_embedding[j]
            other = tail_embedding[k]
            dist_squared = rdist(current, other)
            attractive_force = comparison_utils.attractive_force(
                normalized,
                dist_squared,
                a,
                b,
                edge_weight=1 # Don't scale by weight since we sample according to weight distribution
            )

            for d in range(dim):
                # ANDREW - tsne doesn't do grad clipping
                attraction_d = clip(attractive_force * (current[d] - other[d]))
                current[d] += attraction_d * lr
                if sym_attraction:
                    other[d] -= attraction_d * lr

            epoch_of_next_sample[i] += epochs_per_sample[i]

            # ANDREW - This accounts for the (1 - w(x, y)) in the repulsive grad coefficient
            # It's the same trick as the proportional sampling for the attractive force
            # ...I don't fully understand how this code performs that function
            # FIXME - is this actually changing the sample rate based on 
            #         the (1 - w(current, other)), where 'other' is the randomly
            #         chosen tail embedding?
            #       - It seems to be doing it based on an arbitrary averaging effect
            n_neg_samples = int(
                (i_epoch - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            # ANDREW - UMAP performs multiple repulsive actions for each attractive one!
            # t-SNE, however, gathers the sum of repulsive forces and the sum of attractive forces
            for p in range(n_neg_samples):
                # ANDREW - Picks random vertex from ENTIRE graph and calculates repulsive force
                k = tau_rand_int(rng_state) % n_vertices
                if j == k:
                    continue
                other = tail_embedding[k]
                dist_squared = rdist(current, other)
                repulsive_force, _ = comparison_utils.repulsive_force(
                    normalized,
                    dist_squared,
                    a,
                    b,
                    cell_size=1,
                    average_weight=0, # Don't scale by weight since we sample according to weight distribution
                    Z=0 # don't collect
                )

                for d in range(dim):
                    # ANDREW - tSNE doesn't do gradient clipping
                    if repulsive_force > 0.0:
                        repulsion_d = clip(repulsive_force * (current[d] - other[d]))
                    else:
                        repulsion_d = 4.0
                    current[d] += repulsion_d * lr

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )

    return grads



def optimize_uniformly(
    normalized,
    sym_attraction,
    head_embedding,
    tail_embedding,
    head,
    tail,
    weights,
    grads,
    n_vertices,
    average_weight,
    epochs_per_sample,
    a,
    b,
    rng_state,
    dim,
    lr,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    i_epoch,
):
    attraction = np.zeros_like(head_embedding)
    repulsion = np.zeros_like(head_embedding)
    Z = 0.0

    for i in numba.prange(epochs_per_sample.shape[0]):
        # Gets one of the knn in HIGH-DIMENSIONAL SPACE relative to the sample point
        j = head[i]
        k = tail[i]

        # ANDREW - optimize positive force for each edge
        current = head_embedding[j]
        other = tail_embedding[k]
        dist_squared = rdist(current, other)
        attractive_force = comparison_utils.attractive_force(
            normalized,
            dist_squared,
            a,
            b,
            weights[i]
        )

        for d in range(dim):
            grad_d = clip(attractive_force * (current[d] - other[d]))
            attraction[j, d] += grad_d
            if sym_attraction:
                attraction[k, d] -= grad_d

        # ANDREW - Picks random vertex from ENTIRE graph and calculates repulsive force
        # ANDREW - If we are summing the effects of the forces and multiplying them
        #   by the weights appropriately, we only need to alternate symmetrically
        #   between positive and negative forces rather than doing 1 positive
        #   calculation to n negative ones
        k = tau_rand_int(rng_state) % n_vertices
        other = tail_embedding[k]
        dist_squared = rdist(current, other)
        repulsive_force, Z = comparison_utils.repulsive_force(
            normalized,
            dist_squared,
            a,
            b,
            cell_size=1,
            average_weight=average_weight,
            Z=Z
        )

        for d in range(dim):
            if repulsive_force > 0.0:
                grad_d = clip(repulsive_force * (current[d] - other[d]))
            else:
                grad_d = 4.0
            repulsion[j, d] += grad_d

    attraction, repulsion = comparison_utils.grad_scaling(
        normalized,
        attraction,
        repulsion,
        Z
    )

    head_embedding += (attraction + repulsion) * lr
    return grads


def numba_optimize_layout(
    optimize_method,
    normalized,
    sym_attraction,
    momentum,
    head_embedding,
    tail_embedding,
    head,
    tail,
    weights,
    n_epochs,
    n_vertices,
    epochs_per_sample,
    a,
    b,
    rng_state,
    initial_lr=1.0,
    negative_sample_rate=5.0,
    parallel=False,
    verbose=False,
    **kwargs,
):
    """
    FIXME FIXME FIXME
    """
    dim = head_embedding.shape[1]
    grads = np.zeros([n_vertices, dim], dtype=np.float32)

    # ANDREW - perform negative samples x times more often
    #          by making the number of epochs between samples smaller
    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    # Perform weight scaling on high-dimensional relationships
    weights, initial_lr = comparison_utils.p_scaling(
        normalized,
        weights,
        initial_lr
    )

    n_edges = n_vertices * (n_vertices - 1)
    average_weight = np.sum(weights) / float(n_edges)

    single_step_functions = {
        'umap_uniform': optimize_uniformly,
        'umap_sampling': optimize_through_sampling
    }
    single_step = single_step_functions[optimize_method]
    optimize_fn = numba.njit(
        single_step,
        fastmath=True,
        parallel=parallel
    )

    lr = initial_lr
    for i_epoch in range(n_epochs):
        # FIXME - clean this up!!
        grads = optimize_fn(
            normalized,
            sym_attraction,
            head_embedding,
            tail_embedding,
            head,
            tail,
            weights,
            grads,
            n_vertices,
            average_weight,
            epochs_per_sample,
            a,
            b,
            rng_state,
            dim,
            lr,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            epoch_of_next_sample,
            i_epoch,
        )
        lr = initial_lr * (1.0 - (float(i_epoch) / float(n_epochs)))

        if verbose and i_epoch % int(n_epochs / 10) == 0:
            print("\tcompleted ", i_epoch, " / ", n_epochs, "epochs")

    return head_embedding

