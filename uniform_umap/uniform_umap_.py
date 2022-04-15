# Originally written by Leland McInnes <leland.mcinnes@gmail.com>
# Modified by Andrew Draganov <draganovandrew@gmail.com>
#
# License: BSD 3 clause
from __future__ import print_function

import locale
from warnings import warn
import time

from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.neighbors import KDTree

try:
    import joblib
except ImportError:
    # sklearn.externals.joblib is deprecated in 0.21, will be removed in 0.23
    from sklearn.externals import joblib

import numpy as np
import scipy
import scipy.sparse
from scipy.sparse import tril as sparse_tril, triu as sparse_triu
import scipy.sparse.csgraph
import numba

from . import distances as dist
from . import utils
from . import spectral
# from pynndescent import NNDescent
from nndescent.pynndescent_ import NNDescent
import nndescent.distances as pynnd_dist

locale.setlocale(locale.LC_NUMERIC, "C")

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

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
        bandwidth=1.0,
        umap_metric=True,
    ):
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    sigmas = np.zeros(distances.shape[0], dtype=np.float32)

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        # ANDREW - Calculate rho values
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        rho[i] = np.max(non_zero_dists)

        # ANDREW - Calculating sigma values
        for n in range(n_iter):
            psum = 0.0
            for j in range(1, distances.shape[1]):
                if umap_metric:
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

        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if sigmas[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                sigmas[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if sigmas[i] < MIN_K_DIST_SCALE * mean_distances:
                sigmas[i] = MIN_K_DIST_SCALE * mean_distances

    return sigmas, rho


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
    umap_metric=True,
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
                # This is where the rhos are subtracted for the UMAP
                # pseudo distance metric
                # The sigmas are equivalent to those found for tSNE
                if umap_metric:
                    val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigmas[i])))
                else:
                    val = np.exp(-((knn_dists[i, j]) / (sigmas[i])))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val

    return rows, cols, vals


def standardize_neighbors(graph):
    """
    FIXME
    """
    data = graph.data
    head = graph.row
    tail = graph.col
    min_neighbors = np.min(np.unique(tail, return_counts=True)[1])
    new_head, new_tail, new_data = [], [], []
    current_point = 0
    counter = 0
    for edge in range(head.shape[0]):
        if tail[edge] > current_point:
            current_point = tail[edge]
            counter = 0
        if counter < min_neighbors:
            new_head.append(head[edge])
            new_tail.append(tail[edge])
            new_data.append(data[edge])
            counter += 1
    return scipy.sparse.coo_matrix(
        (new_data, (new_head, new_tail)),
        shape=graph.shape
    )

@numba.njit()
def init_transform(indices, weights, embedding):
    """Given indices and weights and an original embeddings
    initialize the positions of new points relative to the
    indices and weights (of their neighbors in the source data).

    Parameters
    ----------
    indices: array of shape (n_new_samples, n_neighbors)
        The indices of the neighbors of each new sample

    weights: array of shape (n_new_samples, n_neighbors)
        The membership strengths of associated 1-simplices
        for each of the new samples.

    embedding: array of shape (n_samples, dim)
        The original embedding of the source data.

    Returns
    -------
    new_embedding: array of shape (n_new_samples, dim)
        An initial embedding of the new sample points.
    """
    result = np.zeros((indices.shape[0], embedding.shape[1]), dtype=np.float32)

    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            for d in range(embedding.shape[1]):
                result[i, d] += weights[i, j] * embedding[indices[i, j], d]

    return result

@numba.njit()
def init_update(current_init, n_original_samples, indices):
    for i in range(n_original_samples, indices.shape[0]):
        n = 0
        for j in range(indices.shape[1]):
            for d in range(current_init.shape[1]):
                if indices[i, j] < n_original_samples:
                    n += 1
                    current_init[i, d] += current_init[indices[i, j], d]
        for d in range(current_init.shape[1]):
            current_init[i, d] /= n

    return



class UniformUmap(BaseEstimator):
    def __init__(
        self,
        n_neighbors=15,
        n_components=2,
        n_epochs=None,
        learning_rate=1.0,
        random_init=False,
        umap_metric=True,
        tsne_symmetrization=False,
        optimize_method='uniform_umap',
        normalized=0,
        euclidean=True,
        sym_attraction=True,
        frob=False,
        numba=False,
        gpu=False,
        num_threads=-1,
        negative_sample_rate=5,
        transform_queue_size=4.0,
        a=None,
        b=None,
        transform_seed=42,
        force_approximation_algorithm=False,
        verbose=False,
        **kwargs
    ):
        self.n_neighbors = n_neighbors
        self.n_epochs = n_epochs
        self.random_init = random_init
        self.n_components = n_components
        self.learning_rate = learning_rate

        # ANDREW - options for flipping between tSNE and UMAP
        self.tsne_symmetrization = tsne_symmetrization
        self.umap_metric = umap_metric
        self.optimize_method = optimize_method
        self.normalized = normalized
        self.euclidean = euclidean
        self.sym_attraction = sym_attraction
        self.frob = frob
        self.numba = numba
        self.gpu = gpu
        self.euclidean = euclidean

        self.negative_sample_rate = negative_sample_rate
        self.transform_queue_size = transform_queue_size
        self.transform_seed = transform_seed
        self.force_approximation_algorithm = force_approximation_algorithm
        self.verbose = verbose

        self.num_threads = num_threads

        if a is None or b is None:
            self.find_ab_params()
        else:
            self.a = a
            self.b = b

    def _validate_parameters(self):
        if not isinstance(self.random_init, bool):
            raise ValueError("init must be a bool")
        if self.negative_sample_rate < 0:
            raise ValueError("negative sample rate must be positive")
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be greater than 1")
        if not isinstance(self.n_components, int):
            if isinstance(self.n_components, str):
                raise ValueError("n_components must be an int")
            if self.n_components % 1 != 0:
                raise ValueError("n_components must be a whole number")
            try:
                # this will convert other types of int (eg. numpy int64)
                # to Python int
                self.n_components = int(self.n_components)
            except ValueError:
                raise ValueError("n_components must be an int")
        if self.n_components < 1:
            raise ValueError("n_components must be greater than 0")
        if self.n_epochs is not None and (
            self.n_epochs < 0 or not isinstance(self.n_epochs, int)
        ):
            raise ValueError("n_epochs must be a nonnegative integer")
        if self.num_threads < -1 or self.num_threads == 0:
            raise ValueError("num_threads must be a postive integer, or -1 (for all cores)")

    def find_ab_params(self):
        """Fit a, b params for the differentiable curve used in lower
        dimensional fuzzy simplicial complex construction. We want the
        smooth curve (from a pre-defined family with simple gradient) that
        best matches an offset exponential decay.
        """

        def curve(x, a, b):
            return 1.0 / (1.0 + a * x ** (2 * b))

        spread = 1
        min_dist = 0.1
        xv = np.linspace(0, spread * 3, 300)
        yv = np.zeros(xv.shape)
        yv[xv < min_dist] = 1.0
        yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
        params, covar = curve_fit(curve, xv, yv)
        self.a = params[0]
        self.b = params[1]

    def fit(self, X):
        X = check_array(X, dtype=np.float32, accept_sparse="csr", order="C")
        self._raw_data = X
        self._validate_parameters()
        if self.verbose:
            print(str(self))

        self._original_n_threads = numba.get_num_threads()
        if self.num_threads > 0 and self.num_threads is not None:
            numba.set_num_threads(self.num_threads)
        else:
            self.num_threads = self._original_n_threads

        if self.verbose:
            print("Constructing fuzzy simplicial set")
        start = time.time()
        self.nearest_neighbors(X)
        self.fuzzy_simplicial_set(X)
        end = time.time()
        if self.verbose:
            print('Calculating high dim similarities took {:.3f} seconds'.format(end - start))

        if self.verbose:
            print(utils.ts(), "Constructing embedding")
        self.optimize_layout()
        if self.verbose:
            print(utils.ts() + " Finished embedding")

        numba.set_num_threads(self._original_n_threads)
        self._input_hash = joblib.hash(self._raw_data)
        return self

    def make_epochs_per_sample(self, weights):
        result = -1.0 * np.ones(weights.shape[0], dtype=np.float32)
        n_samples = self.n_epochs * (weights / weights.max())
        result[n_samples > 0] = float(self.n_epochs) / n_samples[n_samples > 0]
        return result

    def optimize_layout(self):
        self.graph = self.graph.tocoo()
        self.graph.sum_duplicates()
        n_vertices = self.graph.shape[1]

        start = time.time()
        # For smaller datasets we can use more epochs
        # FIXME - reduce slowly?
        if self.graph.shape[0] <= 10000:
            default_epochs = 500
        else:
            default_epochs = 200

        if self.n_epochs is None:
            self.n_epochs = default_epochs

        if self.random_init:
            embedding = np.random.multivariate_normal(
                mean=np.zeros(self.n_components),
                cov=np.eye(self.n_components),
                size=(self.graph.shape[0])
            ).astype(np.float32)
        else:
            # We add a little noise to avoid local minima for optimization to come
            initialisation = spectral.spectral_layout(
                self._raw_data,
                self.graph,
                self.n_components,
            )
            expansion = 10.0 / np.abs(initialisation).max()
            embedding = (initialisation * expansion).astype(np.float32) 
            embedding += np.random.normal(
                scale=0.0001,
                size=[self.graph.shape[0], self.n_components],
            ).astype(np.float32)
        # ANDREW - renormalize initial embedding to be in range [0, 10]
        embedding = (
            10.0
            * (embedding - np.min(embedding, 0))
            / (np.max(embedding, 0) - np.min(embedding, 0))
        ).astype(np.float32, order="C")

        # head and tail here represent the indices of nodes that have edges in high-dim
        # So for each edge e_{ij}, head is low-dim embedding of point i and tail
        #   is low-dim embedding of point j
        neighbor_counts = np.unique(self.graph.col, return_counts=True)[1]

        # get number of epochs that we will optimize each EDGE for
        epochs_per_sample = self.make_epochs_per_sample(self.graph.data)

        if self.verbose:
            print('optimizing layout...')
        # FIXME FIXME -- need to make sure that all numpy matrices are in
        #   'c' format!
        self._optimize_layout(
            embedding,
            embedding,
            neighbor_counts,
            n_vertices,
            epochs_per_sample,
        )

    def _optimize_layout(
            self,
            head_embedding,
            tail_embedding,
            neighbor_counts,
            n_vertices,
            epochs_per_sample,
        ):
        args = {
            'optimize_method': self.optimize_method,
            'normalized': self.normalized,
            'angular': not self.euclidean,
            'sym_attraction': int(self.sym_attraction),
            'frob': int(self.frob),
            'num_threads': self.num_threads,
            'head_embedding': head_embedding,
            'tail_embedding': tail_embedding,
            'head': self.graph.row,
            'tail': self.graph.col,
            'weights': self.graph.data.astype(np.float32),
            'neighbor_counts': neighbor_counts,
            'n_epochs': self.n_epochs,
            'n_vertices': n_vertices,
            'epochs_per_sample': epochs_per_sample,
            'a': self.a,
            'b': self.b,
            'initial_lr': self.learning_rate,
            'negative_sample_rate': self.negative_sample_rate,
            'verbose': int(self.verbose)
        }
        start = time.time()
        if self.gpu:
            if self.optimize_method != 'uniform_umap':
                raise ValueError('GPU optimization can only be performed in the uniform umap setting')
            from optimize_gpu import gpu_opt_wrapper as optimizer
        elif self.numba:
            if self.optimize_method == 'umap':
                from .numba_optimizers.umap import optimize_layout_euclidean as optimizer
            elif self.optimize_method == 'uniform_umap':
                from .numba_optimizers.uniform_umap import uniform_umap_numba_wrapper as optimizer
            else:
                raise ValueError('Numba optimization only works for umap and uniform umap')
        else:
            if self.optimize_method == 'umap':
                from umap_opt_two import umap_opt_wrapper as optimizer
            elif self.optimize_method == 'tsne':
                from tsne_opt_two import tsne_opt_wrapper as optimizer
            elif self.optimize_method == 'uniform_umap':
                from uniform_umap_opt_two import uniform_umap_opt_wrapper as optimizer
            else:
                raise ValueError("Optimization method is unsupported at the current time")
        self.embedding = optimizer(**args)
        end = time.time()
        self.opt_time = end - start
        # FIXME -- make into logger output
        if self.verbose:
            print('Optimization took {:.3f} seconds'.format(self.opt_time))

    def fuzzy_simplicial_set(self, X):
        if self.knn_indices is None or self.knn_dists is None:
            self.nearest_neighbors(X)

        self.knn_dists = self.knn_dists.astype(np.float32)
        self.sigmas, self.rhos = smooth_knn_dist(
            self.knn_dists,
            float(self.n_neighbors),
            umap_metric=self.umap_metric,
        )

        rows, cols, vals = compute_membership_strengths(
            self.knn_indices,
            self.knn_dists,
            self.sigmas,
            self.rhos,
            umap_metric=self.umap_metric,
        )

        self.graph = scipy.sparse.coo_matrix(
            (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
        )
        transpose = self.graph.transpose()
        if not self.tsne_symmetrization:
            prod_matrix = self.graph.multiply(transpose)
            self.graph = self.graph + transpose - prod_matrix
        else:
            self.graph = (self.graph + transpose) / 2

        # Ignore all edges with small likelihood
        self.graph.data[self.graph.data < (self.graph.data.max() / float(self.n_epochs))] = 0.0
        self.graph.eliminate_zeros()


    def nearest_neighbors(self, X):
        n_trees = min(64, 5 + int(round((X.shape[0]) ** 0.5 / 20.0)))
        n_iters = max(5, int(round(np.log2(X.shape[0]))))

        if self.euclidean:
            distance_func = pynnd_dist.euclidean
        else:
            distance_func = pynnd_dist.cosine

        self.knn_search_index = NNDescent(
            X,
            n_neighbors=self.n_neighbors,
            n_trees=n_trees,
            distance_func=distance_func,
            n_iters=n_iters,
            max_candidates=20,
            n_jobs=self.num_threads,
            verbose=self.verbose,
        )
        self.knn_indices, self.knn_dists = self.knn_search_index.neighbor_graph

        if self.verbose:
            print(utils.ts(), "Finished Nearest Neighbor Search")

    def fit_transform(self, X):
        self.fit(X)
        return self.embedding
