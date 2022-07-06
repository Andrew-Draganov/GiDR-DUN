import time

import numba
import numpy as np
import scipy

from . import utils

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf

@numba.njit(
    locals={
        "psum": numba.types.float32,
        "lo": numba.types.float32,
        "mid": numba.types.float32,
        "hi": numba.types.float32,
    },
    fastmath=True,
)
def get_sigmas_and_rhos(
        distances,
        k,
        n_iter=64,
        pseudo_distance=True,
):
    target = np.log2(k)
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    sigmas = np.zeros(distances.shape[0], dtype=np.float32)

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        # ANDREW - Calculate rho values
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= 1:
            rho[i] = non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0
        # ANDREW - Calculating sigma values with binary search
        for n in range(n_iter):
            psum = 0.0
            for j in range(1, distances.shape[1]):
                if pseudo_distance:
                    d = distances[i, j] - rho[i]
                else:
                    d = distances[i, j]

                if d > 0:
                    psum += np.exp(-d / mid)
                else:
                    psum += 1.0

            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        sigmas[i] = mid

    return sigmas, rho


@numba.njit(
    locals={
        "knn_dists": numba.types.float32[:, ::1],
        "sigmas": numba.types.float32[::1],
        "rhos": numba.types.float32[::1],
        "val": numba.types.float32,
    },
    fastmath=True,
)
def get_similarities(
        knn_indices,
        knn_dists,
        sigmas,
        rhos,
        pseudo_distance=True,
):
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            # If applied to an adjacency matrix points shouldn't be similar to themselves.
            if knn_indices[i, j] == i:
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                # ANDREW - this is where the rhos are subtracted for the UMAP
                # pseudo distance metric
                # The sigmas are equivalent to those found for tSNE
                if pseudo_distance:
                    val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))
                else:
                    val = np.exp(-((knn_dists[i, j]) / (sigmas[i])))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    return rows, cols, vals
