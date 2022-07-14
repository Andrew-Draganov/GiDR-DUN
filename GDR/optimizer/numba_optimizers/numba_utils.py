import numpy as np
import math
import numba

@numba.njit("f4(f4, f4, f4)", cache=True)
def clip(val, low, high):
    if val > high:
        return high
    elif val < low:
        return low
    return val

@numba.njit(
    "f4(f4[::1],f4[::1],i4)",
    fastmath=True,
    cache=True,
    nogil=True,
    locals={
        "result": numba.types.float32,
        "diff": numba.types.float32,
    },
)
def sq_euc_dist(x, y, dim):
    result = 0.0
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result

@numba.njit(cache=True, fastmath=True, nogil=True)
def ang_dist(x, y, dim):
    # cosine distance between vectors x and y
    result = 0.0;
    x_len  = 0.0;
    y_len  = 0.0;
    eps = 0.0001;
    for i in range(dim):
        result += x[i] * y[i]
        x_len += x[i] * x[i]
        y_len += y[i] * y[i]
    if x_len < eps or y_len < eps:
        return 1
    return result / math.sqrt(x_len * y_len)

@numba.njit(cache=True, fastmath=True, nogil=True)
def get_lr(initial_lr, i_epoch, n_epochs, amplify_grads): 
    if amplify_grads == 1:
        return initial_lr
    return initial_lr * (1.0 - float(i_epoch) / (n_epochs))

def print_status(i_epoch, n_epochs):
    print_rate = n_epochs / 10
    if i_epoch % print_rate == 0:
        print("Completed %d / %d epochs" % (i_epoch, n_epochs))

@numba.njit(cache=True, fastmath=True, nogil=True)
def get_avg_weight(weights, n_edges):
    average_weight = 0.0
    for i in range(n_edges):
        average_weight += weights[i]
    average_weight /= n_edges
    return average_weight

@numba.njit(
    "f4(f4,f4,f4)",
    fastmath=True,
    cache=True,
    nogil=True,
    locals={
        "grad_scalar": numba.types.float32,
    },
)
def umap_attr_scalar(dist, a, b):
    grad_scalar = 2.0 * a * b * pow(dist, b - 1.0)
    grad_scalar /= a * pow(dist, b) + 1.0
    return grad_scalar

@numba.njit(
    "f4(f4,f4,f4)",
    fastmath=True,
    cache=True,
    nogil=True,
    locals={
        "phi_ijZ": numba.types.float32,
    },
)
def umap_rep_scalar(dist, a, b):
    phi_ijZ = 2.0 * b
    phi_ijZ /= (0.001 + dist) * (a * pow(dist, b) + 1)
    return phi_ijZ

@numba.njit(
    "f4(f4,f4,f4)",
    fastmath=True,
    cache=True,
    nogil=True,
)
def kernel_function(dist, a, b):
    if b <= 1:
        return 1 / (1 + a * pow(dist, b))
    return pow(dist, b - 1) / (1 + a * pow(dist, b))

@numba.njit(
    "f4(f4,f4)",
    fastmath=True,
    cache=True,
    nogil=True,
    locals={
        "grad_scalar": numba.types.float32,
    },
)
def kl_attr_force(p, q):
    return p * q

@numba.njit(
    "f4(i4,f4,f4)",
    fastmath=True,
    cache=True,
    nogil=True,
)
def kl_rep_force(normalized, q, avg_weight):
    if normalized:
        return q * q

    # Realistically, we should use the actual weight on
    #   the edge e_{ik}, but we have not gone through
    #   and calculated this value for each weight. Instead,
    #   we have only calculated them for the nearest neighbors.
    # So we use the average of the nearest neighbor weights 
    #   as a substitute
    return q * (1 - avg_weight)

@numba.njit(
    "f4(i4,f4,f4)",
    fastmath=True,
    cache=True,
    nogil=True,
)
def frob_attr_force(normalized, p, q):
    if normalized:
        return  p * (q * q + 2 * pow(q, 3))
    return p * q * q

@numba.njit(
    "f4(i4,f4)",
    fastmath=True,
    cache=True,
    nogil=True,
)
def frob_rep_force(normalized, q):
    if normalized:
        return pow(q, 3) + 2 * pow(q, 4)
    return pow(q, 3)

@numba.njit(
    "f4(i4,i4,f4,f4,f4,f4)",
    fastmath=True,
    cache=True,
    nogil=True,
    locals={
        "grad_scalar": numba.types.float32,
    }
)
def attractive_force_func(
    normalized,
    frob,
    dist,
    a,
    b,
    edge_weight
):
    if normalized or frob:
        q = kernel_function(dist, a, b)
    else:
        q = umap_attr_scalar(dist, a, b)

    if frob:
        return frob_attr_force(normalized, edge_weight, q)
    else:
        return kl_attr_force(edge_weight, q)

@numba.njit(
    numba.types.UniTuple(numba.float32, 2)(
        numba.int32,
        numba.int32,
        numba.float32,
        numba.float32,
        numba.float32,
        numba.float32
    ),
    fastmath=True,
    cache=True,
    nogil=True,
    locals={
        "grad_scalar": numba.types.float32,
    }
)
def repulsive_force_func(
    normalized,
    frob,
    dist,
    a,
    b,
    avg_weight
):
    if normalized or frob:
        q = kernel_function(dist, a, b)
    else:
        q = umap_rep_scalar(dist, a, b)

    if frob:
        force = frob_rep_force(normalized, q)
    else:
        force = kl_rep_force(normalized, q, avg_weight)

    return force, q
