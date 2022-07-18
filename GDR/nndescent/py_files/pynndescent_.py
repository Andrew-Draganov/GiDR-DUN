# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 2 clause

from warnings import warn

import numba
import numpy as np
from sklearn.utils import check_random_state, check_array
from sklearn.preprocessing import normalize

import heapq

import GDR.nndescent.py_files.distances as pynnd_dist

from GDR.nndescent.py_files.utils import (
    tau_rand_int,
    tau_rand,
    make_heap,
    deheap_sort,
    new_build_candidates,
    ts,
    simple_heap_push,
    checked_flagged_heap_push,
    has_been_visited,
    mark_visited,
    apply_graph_updates_high_memory,
    apply_graph_updates_low_memory,
    initalize_heap_from_graph_indices,
)

from GDR.nndescent.py_files.rp_trees import (
    make_forest,
    rptree_leaf_array,
    FlatTree,
    denumbaify_tree,
    renumbaify_tree,
    select_side,
    score_linked_tree,
)

update_type = numba.types.List(
    numba.types.List((numba.types.int64, numba.types.int64, numba.types.float64))
)

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

FLOAT32_EPS = np.finfo(np.float32).eps

EMPTY_GRAPH = make_heap(1, 1)


@numba.njit(parallel=True)
def generate_leaf_updates(leaf_block, dist_thresholds, data, dist):
    updates = [[(-1, -1, np.inf)] for i in range(leaf_block.shape[0])]
    for n in numba.prange(leaf_block.shape[0]):
        for i in range(leaf_block.shape[1]):
            p = leaf_block[n, i]
            if p < 0:
                break

            for j in range(i + 1, leaf_block.shape[1]):
                q = leaf_block[n, j]
                if q < 0:
                    break

                d = dist(data[p], data[q])
                if d < dist_thresholds[p] or d < dist_thresholds[q]:
                    updates[n].append((p, q, d))

    return updates


@numba.njit(locals={"d": numba.float32, "p": numba.int32, "q": numba.int32})
def init_rp_tree(data, dist, current_graph, leaf_array):
    n_leaves = leaf_array.shape[0]
    block_size = 65536
    n_blocks = n_leaves // block_size

    for i in range(n_blocks + 1):
        block_start = i * block_size
        block_end = min(n_leaves, (i + 1) * block_size)

        leaf_block = leaf_array[block_start:block_end]
        dist_thresholds = current_graph[1][:, 0]

        updates = generate_leaf_updates(leaf_block, dist_thresholds, data, dist)

        for j in range(len(updates)):
            for k in range(len(updates[j])):
                p, q, d = updates[j][k]

                if p == -1 or q == -1:
                    continue

                checked_flagged_heap_push(
                    current_graph[1][p],
                    current_graph[0][p],
                    current_graph[2][p],
                    d,
                    q,
                    np.uint8(1),
                )
                checked_flagged_heap_push(
                    current_graph[1][q],
                    current_graph[0][q],
                    current_graph[2][q],
                    d,
                    p,
                    np.uint8(1),
                )


@numba.njit(
    fastmath=True,
    locals={"d": numba.float32, "idx": numba.int32, "i": numba.int32},
)
def init_random(n_neighbors, data, heap, dist, rng_state):
    for i in range(data.shape[0]):
        if heap[0][i, 0] < 0.0:
            for j in range(n_neighbors - np.sum(heap[0][i] >= 0.0)):
                idx = np.abs(tau_rand_int(rng_state)) % data.shape[0]
                d = dist(data[idx], data[i])
                checked_flagged_heap_push(
                    heap[1][i], heap[0][i], heap[2][i], d, idx, np.uint8(1)
                )
    return


@numba.njit(parallel=True)
def generate_graph_updates(
    new_candidate_block, old_candidate_block, dist_thresholds, data, dist
):

    block_size = new_candidate_block.shape[0]
    updates = [[(-1, -1, np.inf)] for i in range(block_size)]
    max_candidates = new_candidate_block.shape[1]

    for i in numba.prange(block_size):
        for j in range(max_candidates):
            p = int(new_candidate_block[i, j])
            if p < 0:
                continue

            for k in range(j, max_candidates):
                q = int(new_candidate_block[i, k])
                if q < 0:
                    continue

                d = dist(data[p], data[q])
                if d <= dist_thresholds[p] or d <= dist_thresholds[q]:
                    updates[i].append((p, q, d))

            for k in range(max_candidates):
                q = int(old_candidate_block[i, k])
                if q < 0:
                    continue

                d = dist(data[p], data[q])
                if d <= dist_thresholds[p] or d <= dist_thresholds[q]:
                    updates[i].append((p, q, d))

    return updates


@numba.njit()
def process_candidates(
    data,
    dist,
    current_graph,
    new_candidate_neighbors,
    old_candidate_neighbors,
    n_blocks,
    block_size,
    n_threads,
):
    c = 0
    n_vertices = new_candidate_neighbors.shape[0]
    for i in range(n_blocks + 1):
        block_start = i * block_size
        block_end = min(n_vertices, (i + 1) * block_size)

        new_candidate_block = new_candidate_neighbors[block_start:block_end]
        old_candidate_block = old_candidate_neighbors[block_start:block_end]

        dist_thresholds = current_graph[1][:, 0]

        updates = generate_graph_updates(
            new_candidate_block, old_candidate_block, dist_thresholds, data, dist
        )

        c += apply_graph_updates_low_memory(current_graph, updates, n_threads)

    return c


@numba.njit()
def _nn_descent(
    current_graph,
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    dist=pynnd_dist.euclidean,
    n_iters=10,
    delta=0.001,
    verbose=False,
):
    n_vertices = data.shape[0]
    block_size = 16384
    n_blocks = n_vertices // block_size
    n_threads = numba.get_num_threads()

    for n in range(n_iters):
        if verbose:
            print("\t", n + 1, " / ", n_iters)

        (new_candidate_neighbors, old_candidate_neighbors) = new_build_candidates(
            current_graph, max_candidates, rng_state, n_threads
        )

        c = process_candidates(
            data,
            dist,
            current_graph,
            new_candidate_neighbors,
            old_candidate_neighbors,
            n_blocks,
            block_size,
            n_threads,
        )

        if c <= delta * n_neighbors * data.shape[0]:
            if verbose:
                print("\tStopping threshold met -- exiting after", n + 1, "iterations")
            return


@numba.njit()
def nn_descent(
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    dist=pynnd_dist.euclidean,
    n_iters=10,
    delta=0.001,
    init_graph=EMPTY_GRAPH,
    rp_tree_init=True,
    leaf_array=None,
    verbose=False,
):

    if init_graph[0].shape[0] == 1:  # EMPTY_GRAPH
        current_graph = make_heap(data.shape[0], n_neighbors)
        if rp_tree_init:
            init_rp_tree(data, dist, current_graph, leaf_array)
        init_random(n_neighbors, data, current_graph, dist, rng_state)
    elif (
        init_graph[0].shape[0] == data.shape[0]
        and init_graph[0].shape[1] == n_neighbors
    ):
        current_graph = init_graph
    else:
        raise ValueError("Invalid initial graph specified!")

    _nn_descent(
        current_graph,
        data,
        n_neighbors,
        rng_state,
        max_candidates=max_candidates,
        dist=dist,
        n_iters=n_iters,
        delta=delta,
        verbose=verbose,
    )

    return deheap_sort(current_graph)


@numba.njit(parallel=True)
def diversify(indices, distances, data, dist, rng_state, prune_probability=1.0):

    for i in numba.prange(indices.shape[0]):

        new_indices = [indices[i, 0]]
        new_distances = [distances[i, 0]]
        for j in range(1, indices.shape[1]):
            if indices[i, j] < 0:
                break

            flag = True
            for k in range(len(new_indices)):

                c = new_indices[k]

                d = dist(data[indices[i, j]], data[c])
                if new_distances[k] > FLOAT32_EPS and d < distances[i, j]:
                    if tau_rand(rng_state) < prune_probability:
                        flag = False
                        break

            if flag:
                new_indices.append(indices[i, j])
                new_distances.append(distances[i, j])

        for j in range(indices.shape[1]):
            if j < len(new_indices):
                indices[i, j] = new_indices[j]
                distances[i, j] = new_distances[j]
            else:
                indices[i, j] = -1
                distances[i, j] = np.inf

    return indices, distances


@numba.njit(parallel=True)
def diversify_csr(
    graph_indptr,
    graph_indices,
    graph_data,
    source_data,
    dist,
    rng_state,
    prune_probability=1.0,
):
    n_nodes = graph_indptr.shape[0] - 1

    for i in numba.prange(n_nodes):

        current_indices = graph_indices[graph_indptr[i] : graph_indptr[i + 1]]
        current_data = graph_data[graph_indptr[i] : graph_indptr[i + 1]]

        order = np.argsort(current_data)
        retained = np.ones(order.shape[0], dtype=np.int8)

        for idx in range(1, order.shape[0]):
            j = order[idx]
            for k in range(idx):
                l = order[k]
                if retained[l] == 1:

                    d = dist(
                        source_data[current_indices[j]], source_data[current_indices[k]]
                    )
                    if current_data[l] > FLOAT32_EPS and d < current_data[j]:
                        if tau_rand(rng_state) < prune_probability:
                            retained[j] = 0
                            break

        for idx in range(order.shape[0]):
            j = order[idx]
            if retained[j] == 0:
                graph_data[graph_indptr[i] + j] = 0

    return


@numba.njit(parallel=True)
def degree_prune_internal(indptr, data, max_degree=20):
    for i in numba.prange(indptr.shape[0] - 1):
        row_data = data[indptr[i] : indptr[i + 1]]
        if row_data.shape[0] > max_degree:
            cut_value = np.sort(row_data)[max_degree]
            for j in range(indptr[i], indptr[i + 1]):
                if data[j] > cut_value:
                    data[j] = 0.0

    return


def degree_prune(graph, max_degree=20):
    """Prune the k-neighbors graph back so that nodes have a maximum
    degree of ``max_degree``.

    Parameters
    ----------
    graph: sparse matrix
        The adjacency matrix of the graph

    max_degree: int (optional, default 20)
        The maximum degree of any node in the pruned graph

    Returns
    -------
    result: sparse matrix
        The pruned graph.
    """
    degree_prune_internal(graph.indptr, graph.data, max_degree)
    graph.eliminate_zeros()
    return graph

class NNDescent(object):
    """NNDescent for fast approximate nearest neighbor queries. NNDescent is
    very flexible and supports a wide variety of distances, including
    non-metric distances. NNDescent also scales well against high dimensional
    graph_data in many cases. This implementation provides a straightfoward
    interface, with access to some tuning parameters.

    Parameters
    ----------
    data: array os shape (n_samples, n_features)
        The training graph_data set to find nearest neighbors in.

    n_neighbors: int (optional, default=30)
        The number of neighbors to use in k-neighbor graph graph_data structure
        used for fast approximate nearest neighbor search. Larger values
        will result in more accurate search results at the cost of
        computation time.

    n_trees: int (optional, default=None)
        This implementation uses random projection forests for initializing the index
        build process. This parameter controls the number of trees in that forest. A
        larger number will result in more accurate neighbor computation at the cost
        of performance. The default of None means a value will be chosen based on the
        size of the graph_data.

    leaf_size: int (optional, default=None)
        The maximum number of points in a leaf for the random projection trees.
        The default of None means a value will be chosen based on n_neighbors.

    pruning_degree_multiplier: float (optional, default=1.5)
        How aggressively to prune the graph. Since the search graph is undirected
        (and thus includes nearest neighbors and reverse nearest neighbors) vertices
        can have very high degree -- the graph will be pruned such that no
        vertex has degree greater than
        ``pruning_degree_multiplier * n_neighbors``.

    diversify_prob: float (optional, default=1.0)
        The search graph get "diversified" by removing potentially unnecessary
        edges. This controls the volume of edges removed. A value of 0.0 ensures
        that no edges get removed, and larger values result in significantly more
        aggressive edge removal. A value of 1.0 will prune all edges that it can.

    n_search_trees: int (optional, default=1)
        The number of random projection trees to use in initializing searching or
        querying.

        .. deprecated:: 0.5.5

    tree_init: bool (optional, default=True)
        Whether to use random projection trees for initialization.

    random_state: int, RandomState instance or None, optional (default: None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    algorithm: string (optional, default='standard')
        This implementation provides an alternative algorithm for
        construction of the k-neighbors graph used as a search index. The
        alternative algorithm can be fast for large ``n_neighbors`` values.
        The``'alternative'`` algorithm has been deprecated and is no longer
        available.

    max_candidates: int (optional, default=20)
        Internally each "self-join" keeps a maximum number of candidates (
        nearest neighbors and reverse nearest neighbors) to be considered.
        This value controls this aspect of the algorithm. Larger values will
        provide more accurate search results later, but potentially at
        non-negligible computation cost in building the index. Don't tweak
        this value unless you know what you're doing.

    n_iters: int (optional, default=None)
        The maximum number of NN-descent iterations to perform. The
        NN-descent algorithm can abort early if limited progress is being
        made, so this only controls the worst case. Don't tweak
        this value unless you know what you're doing. The default of None means
        a value will be chosen based on the size of the graph_data.

    delta: float (optional, default=0.001)
        Controls the early abort due to limited progress. Larger values
        will result in earlier aborts, providing less accurate indexes,
        and less accurate searching. Don't tweak this value unless you know
        what you're doing.

    n_jobs: int or None, optional (default=None)
        The number of parallel jobs to run for neighbors index construction.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    verbose: bool (optional, default=False)
        Whether to print status graph_data during the computation.
    """

    def __init__(
        self,
        data,
        n_neighbors=30,
        n_trees=None,
        leaf_size=None,
        pruning_degree_multiplier=1.5,
        diversify_prob=1.0,
        distance_func=pynnd_dist.euclidean,
        n_search_trees=1,
        init_graph=None,
        random_state=None,
        max_candidates=None,
        n_iters=None,
        delta=0.001,
        n_jobs=None,
        verbose=False,
    ):

        if n_trees is None:
            n_trees = 5 + int(round((data.shape[0]) ** 0.25))
            n_trees = min(32, n_trees)  # Only so many trees are useful
        if n_iters is None:
            n_iters = max(5, int(round(np.log2(data.shape[0]))))

        self.n_trees = n_trees
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.prune_degree_multiplier = pruning_degree_multiplier
        self.diversify_prob = diversify_prob
        self.n_search_trees = n_search_trees
        self.max_candidates = max_candidates
        self.n_iters = n_iters
        self.delta = delta
        self.dim = data.shape[1]
        self.n_jobs = n_jobs
        self.verbose = verbose

        data = check_array(data, dtype=np.float32, accept_sparse="csr", order="C")
        self._raw_data = data

        self.random_state = random_state
        current_random_state = check_random_state(self.random_state)

        self._distance_func = distance_func
        self._distance_correction = None
        self._distance_correction = np.sqrt

        self.rng_state = current_random_state.randint(INT32_MIN, INT32_MAX, 3).astype(
            np.int64
        )
        self.search_rng_state = current_random_state.randint(
            INT32_MIN, INT32_MAX, 3
        ).astype(np.int64)
        # Warm up the rng state
        # ANDREW -- what does this do?...
        for i in range(10):
            _ = tau_rand_int(self.search_rng_state)

        if verbose:
            print(ts(), "Building RP forest with", str(n_trees), "trees")
        self._rp_forest = make_forest(
            data,
            n_neighbors,
            n_trees,
            leaf_size,
            self.rng_state,
            current_random_state,
            self.n_jobs,
        )
        leaf_array = rptree_leaf_array(self._rp_forest)

        if self.max_candidates is None:
            effective_max_candidates = min(60, self.n_neighbors)
        else:
            effective_max_candidates = self.max_candidates

        # Set threading constraints
        self._original_num_threads = numba.get_num_threads()
        if self.n_jobs != -1 and self.n_jobs is not None:
            numba.set_num_threads(self.n_jobs)

        if init_graph is None:
            _init_graph = EMPTY_GRAPH
        else:
            if init_graph.shape[0] != self._raw_data.shape[0]:
                raise ValueError("Init graph size does not match dataset size!")
            _init_graph = make_heap(init_graph.shape[0], self.n_neighbors)
            _init_graph = initalize_heap_from_graph_indices(
                _init_graph, init_graph, data, self._distance_func
            )

        if verbose:
            print(ts(), "NN descent for", str(n_iters), "iterations")

        self._neighbor_graph = nn_descent(
            self._raw_data,
            self.n_neighbors,
            self.rng_state,
            effective_max_candidates,
            self._distance_func,
            self.n_iters,
            self.delta,
            rp_tree_init=True,
            init_graph=_init_graph,
            leaf_array=leaf_array,
            verbose=verbose,
        )

        if np.any(self._neighbor_graph[0] < 0):
            warn(
                "Failed to correctly find n_neighbors for some samples."
                "Results may be less than ideal. Try re-running with"
                "different parameters."
            )

        numba.set_num_threads(self._original_num_threads)

    def __getstate__(self):
        result = self.__dict__.copy()
        if hasattr(self, "_rp_forest"):
            del result["_rp_forest"]
        result["_search_forest"] = tuple(
            [denumbaify_tree(tree) for tree in self._search_forest]
        )
        return result

    def __setstate__(self, d):
        self.__dict__ = d
        self._search_forest = tuple(
            [renumbaify_tree(tree) for tree in d["_search_forest"]]
        )

    @property
    def neighbor_graph(self):
        if self._distance_correction is not None:
            result = {
                '_knn_indices': self._neighbor_graph[0].copy(),
                '_knn_dists': self._distance_correction(self._neighbor_graph[1]),
            }
        else:
            result = {
                '_knn_indices': self._neighbor_graph[0].copy(),
                '_knn_dists': self._neighbor_graph[1].copy()
            }

        return result
