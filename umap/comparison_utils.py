import numba
import numpy as np

###################
##### KERNELS #####
###################

@numba.njit()
def umap_attraction_grad(dist_squared, a, b):
    grad_scalar = 0.0
    if dist_squared > 0.0:
        grad_scalar = -2.0 * a * b * pow(dist_squared, b - 1.0)
        grad_scalar /= a * pow(dist_squared, b) + 1.0
    return grad_scalar

@numba.njit()
def umap_repulsion_grad(dist_squared, a, b):
    phi_ijZ = 0.0
    if dist_squared > 0.0:
        phi_ijZ = 2.0 * b
        phi_ijZ /= (0.001 + dist_squared) * (a * pow(dist_squared, b) + 1)
    return phi_ijZ

@numba.njit()
def kernel_function(dist_squared, a, b):
    return 1 / (1 + a * pow(dist_squared, b))

###################
##### WEIGHTS #####
###################

@numba.njit()
def umap_p_scaling(
        weights,
        initial_lr
    ):
    return weights, initial_lr

@numba.njit()
def tsne_p_scaling(
        weights,
        initial_lr
    ):
    # FIXME - add early exaggeration!!
    return weights / np.sum(weights), initial_lr * 200

@numba.njit()
def p_scaling(
        normalized,
        weights,
        initial_lr
    ):
    if not normalized:
        # umap doesn't scale P_ij weights
        return umap_p_scaling(weights, initial_lr)
    return tsne_p_scaling(weights, initial_lr)


@numba.njit()
def umap_repulsive_force(
        dist_squared,
        a,
        b,
        cell_size,
        average_weight,
    ):
    kernel = umap_repulsion_grad(dist_squared, a, b)
    # ANDREW - Using average_weight is a lame approximation
    #        - Realistically, we should use the actual weight on
    #          the edge e_{ik}, but the coo_matrix is not
    #          indexable. So we assume the differences cancel out over
    #          enough iterations
    repulsive_force = cell_size * kernel * (1 - average_weight)
    return repulsive_force, 0.0

@numba.njit()
def tsne_repulsive_force(
        dist_squared,
        a,
        b,
        cell_size,
        Z
    ):
    kernel = kernel_function(dist_squared, a, b)
    Z += cell_size * kernel # Collect the q_ij's contributions into Z
    repulsive_force = cell_size * kernel * kernel
    return repulsive_force, Z

@numba.njit()
def attractive_force(
        normalized,
        dist_squared,
        a,
        b,
        edge_weight
    ):
    if not normalized:
        edge_force = umap_attraction_grad(dist_squared, a, b)
    else:
        edge_force = kernel_function(dist_squared, a, b)

    # FIXME FIXME FIXME
    # This does NOT work with parallel=True
    return edge_force * edge_weight

@numba.njit()
def repulsive_force(
        normalized,
        dist_squared,
        a,
        b,
        cell_size,
        average_weight,
        Z
    ):
    if not normalized:
        return umap_repulsive_force(
            dist_squared,
            a,
            b,
            cell_size,
            average_weight
        )
    return tsne_repulsive_force(
        dist_squared,
        a,
        b,
        cell_size,
        Z
    )

@numba.njit()
def umap_grad_scaling(attraction, repulsion):
    return attraction, repulsion

@numba.njit()
def tsne_grad_scaling(attraction, repulsion, Z):
    repulsion *= - 4 / Z
    attraction *= 4
    return attraction, repulsion

@numba.njit()
def grad_scaling(
        normalized,
        attraction,
        repulsion,
        Z,
    ):
    if not normalized:
        return umap_grad_scaling(attraction, repulsion)
    return tsne_grad_scaling(attraction, repulsion, Z)


