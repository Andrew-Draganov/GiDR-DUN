import time

import numba
import numpy as np
import scipy

from . import utils
from nndescent.py_files.pynndescent_ import NNDescent
import nndescent.py_files.distances as pynnd_dist

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
)  # benchmarking `parallel=True` shows it to *decrease* performance
def smooth_knn_dist(
        distances,
        k,
        n_iter=64,
        local_connectivity=1.0,
        bandwidth=1.0,
        pseudo_distance=True,
):
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    sigmas = np.zeros(distances.shape[0], dtype=np.float32)

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        # ANDREW - Calculate rho values
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (
                            non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0
        # ANDREW - Calculating sigma values
        for n in range(n_iter):
            psum = 0.0
            for j in range(1, distances.shape[1]):
                # ANDREW - when adding option for turning UMAP pseudo distance on/off,
                #   an equivalent change needs to occur here!!
                # FIXME - this if-statement broke the nndescent_umap_test
                #       - it appears that it simply rotates the images around?
                if pseudo_distance:
                    d = distances[i, j] - rho[i]
                else:
                    d = distances[i, j]

                if d > 0:
                    psum += np.exp(-(d / mid))
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

        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if sigmas[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                # ANDREW - this never gets called on mnist
                sigmas[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            # ANDREW - this never gets called on mnist either
            if sigmas[i] < MIN_K_DIST_SCALE * mean_distances:
                sigmas[i] = MIN_K_DIST_SCALE * mean_distances

    return sigmas, rho

def nearest_neighbors(
        X,
        n_neighbors,
        euclidean,
        random_state,
        num_threads=-1,
        verbose=False,
):
    if verbose:
        print(utils.ts(), "Finding Nearest Neighbors")

    # TODO: Hacked values for now
    n_trees = min(64, 5 + int(round((X.shape[0]) ** 0.5 / 20.0)))
    n_iters = max(5, int(round(np.log2(X.shape[0]))))

    if euclidean:
        distance_func = pynnd_dist.euclidean
    else:
        distance_func = pynnd_dist.cosine

    knn_indices, knn_dists = NNDescent(
        X,
        n_neighbors=n_neighbors,
        random_state=random_state,
        n_trees=n_trees,
        distance_func=distance_func,
        n_iters=n_iters,
        max_candidates=20,
        n_jobs=num_threads,
        verbose=verbose,
    ).neighbor_graph

    if verbose:
        print(utils.ts(), "Finished Nearest Neighbor Search")
    return knn_indices, knn_dists


@numba.njit(
    locals={
        "knn_dists": numba.types.float32[:, ::1],
        "sigmas": numba.types.float32[::1],
        "rhos": numba.types.float32[::1],
        "val": numba.types.float32,
    },
    # FIXME FIXME
    # parallel=True,
    fastmath=True,
)
def compute_membership_strengths(
        knn_indices,
        knn_dists,
        sigmas,
        rhos,
        bipartite=False,
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
            # If applied to an incidence matrix (or bipartite) then the row and column indices are different.
            if (bipartite == False) & (knn_indices[i, j] == i):
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


def fuzzy_simplicial_set(
        X,
        n_neighbors,
        random_state,
        knn_indices,
        knn_dists,
        local_connectivity=1.0,
        verbose=False,
        pseudo_distance=True,
        euclidean=True,
        tsne_symmetrization=False,
        gpu=False,
):
    # FIXME -- We shouldn't have this and the exact same thing in the .fit() method
    knn_dists = knn_dists.astype(np.float32)

    if not gpu:
        sigmas, rhos = smooth_knn_dist(
            knn_dists,
            float(n_neighbors),
            local_connectivity=float(local_connectivity),
            pseudo_distance=pseudo_distance,
        )

        rows, cols, vals = compute_membership_strengths(
            knn_indices,
            knn_dists,
            sigmas,
            rhos,
            pseudo_distance=pseudo_distance,
        )
    else:
        from graph_weights_build import graph_weights
        n_points = int(X.shape[0])
        sigmas = np.zeros([n_points], dtype=np.float32, order='c')
        rhos = np.zeros([n_points], dtype=np.float32, order='c')
        rows = np.zeros([n_points * n_neighbors], dtype=np.int32, order='c')
        cols = np.zeros([n_points * n_neighbors], dtype=np.int32, order='c')
        vals = np.zeros([n_points * n_neighbors], dtype=np.float32, order='c')
        dists = np.zeros([n_points * n_neighbors], dtype=np.float32, order='c')
        graph_weights(
            sigmas,
            rhos,
            rows,
            cols,
            vals,
            dists,
            knn_indices.astype(np.int32),
            knn_dists,
            int(n_neighbors),
            0, # FIXME -- this was the return_dists variable, but it is always zero. Remove from C/Cuda files
            float(local_connectivity),
            int(pseudo_distance)
        )

    result = scipy.sparse.coo_matrix(
        (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
    )
    result.eliminate_zeros()

    # UMAP symmetrization:
    # Symmetrized = A + A^T - pointwise_mul(A, A^T)
    # TSNE symmetrization:
    # Symmetrized = (A + A^T) / 2
    transpose = result.transpose()
    if not tsne_symmetrization:
        prod_matrix = result.multiply(transpose)
        result = result + transpose - prod_matrix
    else:
        result = (result + transpose) / 2

    result.eliminate_zeros()

    return result, sigmas, rhos
