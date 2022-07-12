# Originally written by Leland McInnes <leland.mcinnes@gmail.com>
# Modified by Andrew Draganov <draganovandrew@gmail.com>
#
# License: BSD 3 clause
from __future__ import print_function

import locale
import time

from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array
import numpy as np
import scipy
import numba

from . import distances as dist
from . import utils
from . import spectral
from .graph_weights import get_similarities, get_sigmas_and_rhos
from GDR.nndescent.py_files.pynndescent_ import NNDescent
import GDR.nndescent.py_files.distances as pynnd_dist

locale.setlocale(locale.LC_NUMERIC, "C")
INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

def make_epochs_per_sample(weights, n_epochs):
    """
    UMAP legacy function for finding number of epochs between sample optimizations
    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float32)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / n_samples[n_samples > 0]
    return result

def find_ab_params(spread, min_dist):
    """
    UMAP legacy function to find a,b parameters in low-dimensional similarity measure
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]

class GradientDR(BaseEstimator):
    def __init__(
            self,
            n_neighbors=15,
            dim=2,
            n_epochs=None,
            learning_rate=1.0,
            random_init=False,
            pseudo_distance=True,
            tsne_symmetrization=False,
            optimize_method='gdr',
            normalized=0,
            angular=False,
            sym_attraction=True,
            frob=False,
            cython=False,
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
        self.dim = dim
        self.learning_rate = learning_rate

        # ANDREW - options for flipping between tSNE and UMAP
        self.tsne_symmetrization = tsne_symmetrization
        self.pseudo_distance = pseudo_distance
        self.optimize_method = optimize_method
        self.normalized = normalized
        self.angular = angular
        self.sym_attraction = sym_attraction
        self.frob = frob
        self.cython = cython
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
        if not isinstance(self.dim, int):
            if isinstance(self.dim, str):
                raise ValueError("dim must be an int")
            if self.dim % 1 != 0:
                raise ValueError("dim must be a whole number")
            try:
                # this will convert other types of int (eg. numpy int64)
                # to Python int
                self.dim = int(self.dim)
            except ValueError:
                raise ValueError("dim must be an int")
        if self.dim < 1:
            raise ValueError("dim must be greater than 0")
        if self.n_epochs is not None and (
                self.n_epochs < 0 or not isinstance(self.n_epochs, int)
        ):
            raise ValueError("n_epochs must be a nonnegative integer")

        if self.num_threads < -1 or self.num_threads == 0:
            raise ValueError("num_threads must be a postive integer, or -1 (for all cores)")

    def set_num_threads(self):
        if not self.cython and not self.gpu:
            self._original_n_threads = numba.get_num_threads()
            if self.num_threads > 0 and self.num_threads is not None:
                numba.set_num_threads(self.num_threads)
            else:
                self.num_threads = self._original_n_threads

    def reset_num_threads(self):
        if not self.cython and not self.gpu:
            numba.set_num_threads(self._original_n_threads)

    def get_nearest_neighbors(self, X):
        if self.verbose:
            print(utils.ts(), "Finding Nearest Neighbors")
        # Only run GPU nearest neighbors if the dataset is small enough
        # It is exact, so it scales at n^2 vs. NNDescent's nlogn
        if self.gpu and X.shape[0] < 100000 and X.shape[1] < 30000:
            print("doing GPU KNN")
            # FIXME -- make this a try-except. If cudf/cupy didn't install, run on cpu
            from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
            import cudf
            knn_cuml = cuNearestNeighbors(n_neighbors=self.n_neighbors)
            cu_X = cudf.DataFrame(X)
            knn_cuml.fit(cu_X)
            dists, inds = knn_cuml.kneighbors(X)
            self._knn_dists = np.reshape(dists, [X.shape[0], self.n_neighbors])
            self._knn_indices = np.reshape(inds, [X.shape[0], self.n_neighbors])
        else:
            # Legacy values from UMAP implementation
            n_trees = min(64, 5 + int(round((X.shape[0]) ** 0.5 / 20.0)))
            n_iters = max(5, int(round(np.log2(X.shape[0]))))

            if self.angular:
                distance_func = pynnd_dist.cosine
            else:
                distance_func = pynnd_dist.euclidean

            self._knn_indices, self._knn_dists = NNDescent(
                X,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
                n_trees=n_trees,
                distance_func=distance_func,
                n_iters=n_iters,
                max_candidates=20,
                n_jobs=self.num_threads,
                verbose=self.verbose,
            ).neighbor_graph

        if self.verbose:
            print(utils.ts(), "Finished Nearest Neighbor Search")

    def compute_P_matrix(self, X):
        self._knn_dists = self._knn_dists.astype(np.float32)

        # Calculate likelihood of point x_j with respect to point_i
        if not self.gpu:
            # Get umap p(x_j | x_i) constants
            sigmas, rhos = get_sigmas_and_rhos(
                self._knn_dists,
                float(self.n_neighbors),
                pseudo_distance=self.pseudo_distance,
            )

            # Calculate weights in the similarity graph
            rows, cols, vals = get_similarities(
                self._knn_indices,
                self._knn_dists,
                sigmas,
                rhos,
                pseudo_distance=self.pseudo_distance,
            )
        else:
            from gpu_graph_build import graph_weights
            n_points = int(X.shape[0])
            # Initialize memory that will be passed to Cuda as pointers
            # The Cuda functions will then fill these with the appropriate values
            sigmas = np.zeros([n_points], dtype=np.float32, order='c')
            rhos = np.zeros([n_points], dtype=np.float32, order='c')
            rows = np.zeros([n_points * self.n_neighbors], dtype=np.int32, order='c')
            cols = np.zeros([n_points * self.n_neighbors], dtype=np.int32, order='c')
            vals = np.zeros([n_points * self.n_neighbors], dtype=np.float32, order='c')
            dists = np.zeros([n_points * self.n_neighbors], dtype=np.float32, order='c')
            graph_weights(
                sigmas,
                rhos,
                rows,
                cols,
                vals,
                dists,
                self._knn_indices.astype(np.int32),
                self._knn_dists,
                int(self.n_neighbors),
                int(self.pseudo_distance)
            )

        # Put p_{i|j} similarities into a sparse matrix format
        result = scipy.sparse.coo_matrix(
            (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
        )
        result.eliminate_zeros()

        # Symmetrize the similarities
        # UMAP symmetrization:
        # Symmetrized = A + A^T - pointwise_mul(A, A^T)
        # TSNE symmetrization:
        # Symmetrized = (A + A^T) / 2
        transpose = result.transpose()
        if not self.tsne_symmetrization:
            prod_matrix = result.multiply(transpose)
            result = result + transpose - prod_matrix
        else:
            result = (result + transpose) / 2
        result.eliminate_zeros()

        return result

    def fit(self, X):
        """
        FIXME
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
                    (1, self.dim)
                )  # needed to sklearn comparability
                return self, 0

            self.n_neighbors = X.shape[0] - 1

        self.random_state = check_random_state(self.random_state)

        if self.verbose:
            print("Constructing nearest neighbor graph...")
        start = time.time()

        # Get nearest neighbors in high-dimensional space
        self.get_nearest_neighbors(X)

        # Get similarity of high-dimensional points to one another
        self.graph = self.compute_P_matrix(X)

        end = time.time()
        if self.verbose:
            print('Calculating high dim similarities took {:.3f} seconds'.format(end - start))

        # Move around low-dimensional points such that they match the high-dim ones
        self._fit_embed_data(X)

        self.reset_num_threads()

        return self

    def initialize_embedding(self, X):
        if self.random_init:
            embedding = self.random_state.multivariate_normal(
                mean=np.zeros(self.dim),
                cov=np.eye(self.dim),
                size=(self.graph.shape[0])
            ).astype(np.float32)
        else:
            # We add a little noise to avoid local minima for optimization to come
            if X.shape[0] > 100000 and self.verbose:
                print('Doing spectral embedding on large datasets is slow. Consider random initialization.')
            initialisation = spectral.spectral_layout(
                X,
                self.graph,
                self.dim,
                self.random_state,
                metric=dist.euclidean,
            )
            expansion = 10.0 / np.abs(initialisation).max()
            embedding = (initialisation * expansion).astype(np.float32)
            noise_shape = [self.graph.shape[0], self.dim]
            embedding += self.random_state.normal(scale=0.0001, size=noise_shape).astype(np.float32)

        # Renormalize initial embedding to be in range [0, 10]
        embedding -= np.min(embedding, 0)
        embedding /= np.max(embedding) / 10
        self.embedding = embedding.astype(np.float32, order="C")


    def _fit_embed_data(self, X):
        """
        FIXME
        """
        if self.verbose:
            print(utils.ts(), "Constructing embedding")

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
        #        - So for each edge e_{ij} in the nearest neighbor graph,
        #          head is reference to point i and tail is reference to point j
        self.head = self.graph.row
        self.tail = self.graph.col

        # neighbor_counts are used to speed up GPU calculations by standardizing block sizes
        self.neighbor_counts = np.unique(self.tail, return_counts=True)[1].astype(np.long)

        # ANDREW - get number of epochs that we will optimize this EDGE for
        # These are only used in the UMAP algorithm
        self.epochs_per_sample = make_epochs_per_sample(self.graph.data, self.n_epochs)
        self.rng_state = self.random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

        if self.verbose:
            print(utils.ts() + ' Optimizing layout...')
        self._optimize_layout()

        if self.verbose:
            print(utils.ts() + " Finished embedding")


    def _optimize_layout(self):
        # FIXME -- head_embedding and tail_embedding are always the same since 
        #          we don't have separate fit and transform functions
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
            'weights': self.graph.data.astype(np.float32),
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
            if self.optimize_method != 'gdr':
                raise ValueError('GPU optimization can only be performed in the gdr setting')
            from optimize_gpu import gpu_opt_wrapper as optimizer
        elif self.torch:
            if self.optimize_method != 'gdr':
                raise ValueError('PyTorch optimization can only be performed in the gdr setting')
            from .pytorch_optimize import torch_optimize_layout as optimizer
        elif self.cython:
            if self.optimize_method == 'umap':
                from umap_cython import umap_opt_wrapper as optimizer
            elif self.optimize_method == 'tsne':
                from tsne_cython import tsne_opt_wrapper as optimizer
            elif self.optimize_method == 'gdr':
                from gdr_cython import gdr_opt_wrapper as optimizer
            else:
                raise ValueError("Optimization method is unsupported at the current time")
        else:
            if self.optimize_method == 'umap':
                from GDR.optimizer.numba_optimizers import umap_numba_wrapper as optimizer
            elif self.optimize_method == 'gdr':
                from GDR.optimizer.numba_optimizers import gdr_numba_wrapper as optimizer
            else:
                raise ValueError('Numba optimization only works for umap and gdr')
        self.embedding = optimizer(**args)
        end = time.time()
        self.opt_time = end - start
        if self.verbose:
            print('Optimization took {:.3f} seconds'.format(self.opt_time))

    def fit_transform(self, X):
        """
        FIXME
        """
        self.fit(X)
        return self.embedding
