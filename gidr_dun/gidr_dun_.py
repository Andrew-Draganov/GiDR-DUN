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
from . import graph_weights

locale.setlocale(locale.LC_NUMERIC, "C")

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.

    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights ofhow much we wish to sample each 1-simplex.

    n_epochs: int
        The total number of epochs we want to train for.

    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float32)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    return result

def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]

class GidrDun(BaseEstimator):
    def __init__(
            self,
            n_neighbors=15,
            n_components=2,
            n_epochs=None,
            learning_rate=1.0,
            random_init=False,
            pseudo_distance=True,
            tsne_symmetrization=False,
            optimize_method='umap_sampling',
            normalized=0,
            angular=False,
            sym_attraction=True,
            frob=False,
            numba=False,
            torch=False,
            gpu=False,
            amplify_grads=False,
            min_dist=0.1,
            spread=1.0,
            num_threads=-1,
            negative_sample_rate=5,
            a=None,
            b=None,
            random_state=None,
            verbose=False,
    ):
        self.n_neighbors = n_neighbors
        self.random_init = random_init
        self.n_components = n_components
        self.learning_rate = learning_rate

        # ANDREW - options for flipping between tSNE and UMAP
        self.tsne_symmetrization = tsne_symmetrization
        self.pseudo_distance = pseudo_distance
        self.optimize_method = optimize_method
        self.normalized = normalized
        self.angular = angular
        self.sym_attraction = sym_attraction
        self.frob = frob
        self.numba = numba
        self.torch = torch
        self.gpu = gpu
        self.amplify_grads = amplify_grads

        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state
        self.verbose = verbose

        self.num_threads = num_threads

        self.min_dist = min_dist
        self.spread = spread
        if a is None or b is None:
            self.a, self.b = find_ab_params(self.spread, self.min_dist)
        else:
            self.a = a
            self.b = b

        if n_epochs is None:
            if normalized:
                self.n_epochs = 500 # TSNE has weaker gradients and needs more steps to converge
            else:
                self.n_epochs = 200
        else:
            self.n_epochs = n_epochs

    def _validate_parameters(self):
        """ Legacy UMAP parameter validation """
        if not isinstance(self.random_init, bool):
            raise ValueError("init must be a bool")
        if self.negative_sample_rate < 0:
            raise ValueError("negative sample rate must be positive")
        if self.learning_rate < 0.0:
            raise ValueError("learning_rate must be positive")
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

    def set_num_threads(self):
        if self.numba:
            self._original_n_threads = numba.get_num_threads()
            if self.num_threads > 0 and self.num_threads is not None:
                numba.set_num_threads(self.num_threads)
            else:
                self.num_threads = self._original_n_threads

    def reset_num_threads(self):
        if self.numba:
            numba.set_num_threads(self._original_n_threads)


    def fit(self, X):
        """Fit X into an embedded space.

        Optionally use y for supervised dimension reduction.

        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'.
        """

        X = check_array(X, dtype=np.float32, order="C")

        # Handle all the optional arguments, setting default
        self._validate_parameters()
        if self.verbose:
            print(str(self))

        self.set_num_threads()

        # Error check n_neighbors based on data size
        if X.shape[0] <= self.n_neighbors:
            if X.shape[0] == 1:
                self.embedding_ = np.zeros(
                    (1, self.n_components)
                )  # needed to sklearn comparability
                return self, 0

            self.n_neighbors = X.shape[0] - 1

        self.random_state = check_random_state(self.random_state)
        if self.verbose:
            print("Constructing nearest neighbor graph...")

        start = time.time()
        # Only run GPU nearest neighbors if the dataset is small enough
        # It is exact, so it scales at n^2 vs. NNDescent's nlogn
        if self.gpu and X.shape[0] < 100000 and X.shape[1] < 30000:
            print("doing GPU KNN")
            from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
            import cudf
            import cupy as cp
            knn_cuml = cuNearestNeighbors(n_neighbors=self.n_neighbors)
            cu_X = cudf.DataFrame(X)
            knn_cuml.fit(cu_X)
            dists, inds = knn_cuml.kneighbors(cu_X)
            self._knn_dists = np.reshape(dists.to_numpy(), [X.shape[0], self.n_neighbors])
            self._knn_indices = np.reshape(inds.to_numpy(), [X.shape[0], self.n_neighbors])
        else:
            self._knn_indices, self._knn_dists = graph_weights.nearest_neighbors(
                X,
                self.n_neighbors,
                self.angular,
                self.random_state,
                num_threads=self.num_threads,
                verbose=True,
            )

        self.graph = graph_weights.compute_P_matrix(
            X,
            self.n_neighbors,
            self.random_state,
            self._knn_indices,
            self._knn_dists,
            self.verbose,
            pseudo_distance=self.pseudo_distance,
            tsne_symmetrization=self.tsne_symmetrization,
            gpu=self.gpu,
        )
        end = time.time()
        if self.verbose:
            print('Calculating high dim similarities took {:.3f} seconds'.format(end - start))

        if self.verbose:
            print(utils.ts(), "Constructing embedding")

        self._fit_embed_data(X)

        if self.verbose:
            print(utils.ts() + " Finished embedding")

        self.reset_num_threads()

        return self

    def initialize_embedding(self, X):
        if self.random_init or self.gpu:
            embedding = self.random_state.multivariate_normal(
                mean=np.zeros(n_components), cov=np.eye(n_components), size=(graph.shape[0])
            ).astype(np.float32)
        else:
            # We add a little noise to avoid local minima for optimization to come
            initialisation = spectral.spectral_layout(
                X,
                self.graph,
                self.n_components,
                self.random_state,
                metric=dist.euclidean,
            )
            expansion = 10.0 / np.abs(initialisation).max()
            embedding = (initialisation * expansion).astype(np.float32)
            noise_shape = [self.graph.shape[0], self.n_components]
            embedding += self.random_state.normal(scale=0.0001, size=noise_shape).astype(np.float32)

        # ANDREW - renormalize initial embedding to be in range [0, 10]
        embedding -= np.min(embedding, 0)
        embedding /= np.max(embedding) / 10
        self.embedding = embedding.astype(np.float32, order="C")


    def _fit_embed_data(self, X):
        """
        FIXME
        """
        self.graph = self.graph.tocoo()
        self.graph.sum_duplicates()
        start = time.time()

        if self.n_epochs > 10:
            self.graph.data[self.graph.data < (self.graph.data.max() / float(self.n_epochs))] = 0.0
        else:
            self.graph.data[graph.data < (self.graph.data.max() / float(self.default_epochs))] = 0.0
        self.graph.eliminate_zeros()

        self.initialize_embedding(X)

        # ANDREW - head and tail here represent the indices of nodes that have edges in high-dim
        #        - So for each edge e_{ij}, head is low-dim embedding of point i and tail
        #          is low-dim embedding of point j
        self.head = self.graph.row
        self.tail = self.graph.col
        self.neighbor_counts = np.unique(self.tail, return_counts=True)[1].astype(np.long)

        # ANDREW - get number of epochs that we will optimize this EDGE for
        # These are only used in the UMAP algorithm
        self.epochs_per_sample = make_epochs_per_sample(self.graph.data, self.n_epochs)
        self.rng_state = self.random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

        if self.verbose:
            print('optimizing layout...')
        self._optimize_layout()

    def _optimize_layout(self):
        weights = self.graph.data.astype(np.float32)
        args = {
            'optimize_method': self.optimize_method,
            'normalized': self.normalized,
            'angular': self.angular,
            'sym_attraction': int(self.sym_attraction),
            'frob': int(self.frob),
            'num_threads': self.num_threads,
            'amplify_grads': int(self.amplify_grads),
            'head_embedding': self.embedding,
            'tail_embedding': self.embedding,
            'head': self.head,
            'tail': self.tail,
            'weights': weights,
            'neighbor_counts': self.neighbor_counts,
            'n_epochs': self.n_epochs,
            'n_vertices': self.graph.shape[1],
            'epochs_per_sample': self.epochs_per_sample,
            'a': self.a,
            'b': self.b,
            'initial_lr': self.learning_rate,
            'negative_sample_rate': self.negative_sample_rate,
            'rng_state': self.rng_state,
            'verbose': int(self.verbose)
        }
        start = time.time()
        if self.gpu:
            if self.optimize_method != 'gidr_dun':
                raise ValueError('GPU optimization can only be performed in the gidr_dun setting')
            from optimize_gpu import gpu_opt_wrapper as optimizer
        elif self.torch:
            if self.optimize_method != 'gidr_dun':
                raise ValueError('PyTorch optimization can only be performed in the gidr_dun setting')
            from .pytorch_optimize import torch_optimize_layout as optimizer
        elif self.numba:
            if self.optimize_method == 'umap':
                from .numba_optimizers.umap import optimize_layout_euclidean as optimizer
            elif self.optimize_method == 'gidr_dun':
                from .numba_optimizers.gidr_dun import gidr_dun_numba_wrapper as optimizer
            else:
                raise ValueError('Numba optimization only works for umap and gidr_dun')
        else:
            if self.optimize_method == 'umap':
                from umap_opt import umap_opt_wrapper as optimizer
            elif self.optimize_method == 'tsne':
                from tsne_opt import tsne_opt_wrapper as optimizer
            elif self.optimize_method == 'gidr_dun':
                from gidr_dun_opt import gidr_dun_opt_wrapper as optimizer
            else:
                raise ValueError("Optimization method is unsupported at the current time")
        self.embedding = optimizer(**args)
        end = time.time()
        self.opt_time = end - start
        # FIXME -- make into logger output
        if self.verbose:
            print('Optimization took {:.3f} seconds'.format(self.opt_time))


    def fit_transform(self, X):
        """
        FIXME
        """
        self.fit(X)
        return self.embedding
