import numba
import numpy as np

###################
##### KERNELS #####
###################

@numba.njit()
def umap_pos_force_kernel(dist_squared, a, b):
    grad_scalar = 0.0
    if dist_squared > 0.0:
        grad_scalar = -2.0 * a * b * pow(dist_squared, b - 1.0)
        grad_scalar /= a * pow(dist_squared, b) + 1.0
    return grad_scalar

@numba.njit()
def umap_neg_force_kernel(dist_squared, a, b):
    phi_ijZ = 0.0
    if dist_squared > 0.0:
        phi_ijZ = 2.0 * b
        phi_ijZ /= (0.001 + dist_squared) * (a * pow(dist_squared, b) + 1)
    return phi_ijZ

@numba.njit()
def tsne_kernel(dist_squared):
    return 1 / (1 + dist_squared)

@numba.njit()
def pos_force_kernel(kernel_choice, dist_squared, a, b):
    if kernel_choice == 'umap':
        return umap_pos_force_kernel(dist_squared, a, b)
    assert kernel_choice == 'tsne'
    return tsne_kernel(dist_squared)

@numba.njit()
def neg_force_kernel(kernel_choice, dist_squared, a, b):
    if kernel_choice == 'umap':
        return umap_neg_force_kernel(dist_squared, a, b)
    assert kernel_choice == 'tsne'
    return tsne_kernel(dist_squared)

###################
##### WEIGHTS #####
###################

@numba.njit()
def umap_pos_weight_scaling(
        weights,
        initial_alpha
    ):
    return weights, initial_alpha

@numba.njit()
def tsne_pos_weight_scaling(
        weights,
        initial_alpha
    ):
    # FIXME - early exaggeration!!
    return weights / np.sum(weights), initial_alpha * 200

@numba.njit()
def pos_weight_scaling(
        weight_scaling_choice,
        weights,
        initial_alpha
    ):
    if weight_scaling_choice == 'umap':
        # umap doesn't scale P_ij weights
        return umap_pos_weight_scaling(weights, initial_alpha)
    assert weight_scaling_choice == 'tsne'
    return tsne_pos_weight_scaling(weights, initial_alpha)


@numba.njit()
def umap_neg_weight_scaling(
        kernel,
        cell_size,
        average_weight,
    ):
    neg_force = kernel * (1 - average_weight) * cell_size
    return neg_force, 0.0

@numba.njit()
def tsne_neg_weight_scaling(
        kernel,
        cell_size,
        weight_scalar
    ):
    weight_scalar += cell_size * kernel # Collect the q_ij's contributions into Z
    neg_force = cell_size * kernel * kernel
    return neg_force, weight_scalar

@numba.njit()
def neg_weight_scaling(
        weight_scaling_choice,
        kernel,
        cell_size,
        average_weight,
        weight_scalar
    ):
    if weight_scaling_choice == 'umap':
        return umap_neg_weight_scaling(kernel, cell_size, average_weight)
    assert weight_scaling_choice == 'tsne'
    return tsne_neg_weight_scaling(kernel, cell_size, weight_scalar)


@numba.njit()
def umap_total_weight_scaling(pos_grads, neg_grads):
    return pos_grads, neg_grads

@numba.njit()
def tsne_total_weight_scaling(pos_grads, neg_grads, weight_scalar):
    neg_grads *= 4 / weight_scalar
    pos_grads *= 4
    return pos_grads, neg_grads

@numba.njit()
def total_weight_scaling(
        weight_scaling_choice,
        pos_grads,
        neg_grads,
        weight_scalar,
    ):
    if weight_scaling_choice == 'umap':
        return umap_total_weight_scaling(pos_grads, neg_grads)
    assert weight_scaling_choice == 'tsne'
    return tsne_total_weight_scaling(pos_grads, neg_grads, weight_scalar)


