from cuml.manifold.umap import UMAP as cumlUMAP
from cuml.manifold import TSNE as cumlTSNE
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors

import cudf
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class rapids_umap:
    def __init__(self, n_neighbors, n_components, n_epochs, negative_sample_rate, a, b, random_state, verbose):
        self.dr = cumlUMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            n_epochs=n_epochs,
            # learning_rate=params['learning_rate'], #already default to 1.0 in rapids
            #random_init=params['random_init'], #default spectral
            #min_dist=params['min_dist'], #default 0.1
            #spread=params['spread'], #default 1.0
            #set_op_mix_ratio=params['set_op_mix_ratio'], #default 1.0
            #local_connectivity=params['local_connectivity'], #default 1
            #repulsion_strength=params['repulsion_strength'], #default 1.0
            negative_sample_rate=negative_sample_rate,
            #transform_queue_size=params['transform_queue_size'], #default 4
            a=a,
            b=b,
            #hash_input=['hash_input'], #default false
            random_state=random_state,
            verbose=verbose,
            output_type='numpy'
        )

        self.n_neighbors = n_neighbors
        self.verbose = verbose
    
    def fit_transform(self, points):
        # if self.verbose:
        #     print('loading data ...')
        # start = time.time()
        # gdf = cudf.DataFrame()
        # for i in range(points.shape[1]):
        #     gdf['fea%d'%i] = points[:,i]
        # end = time.time()
        # self.opt_time = end -start
        # if self.verbose:
        #     print('Data load took {:.3f} seconds'.format(end - start))

        if self.verbose:
            print("making knn graph...")
        t0 = time.time()
        knn_cuml = cuNearestNeighbors(n_neighbors=self.n_neighbors)
        knn_cuml.fit(points)
        knn_graph_comp = knn_cuml.kneighbors_graph(points)
        t1 = time.time()
        if self.verbose:
            print("knn graph took", t1 - t0)

        if self.verbose:
            print('fitting...')
        start = time.time()
        projection = self.dr.fit_transform(points, knn_graph=knn_graph_comp)
        end = time.time()
        if self.verbose:
            print('CUML UMAP took {:.3f} seconds'.format(end - start))

        self.opt_time = end - start

        del knn_cuml
        del knn_graph_comp
        # del gdf

        return projection

class rapids_tsne:
    def __init__(self, n_components, learning_rate, n_epochs, verbose, random_state, n_neighbors):
        self.dr = cumlTSNE(
            n_components=n_components,
            #perplexity=params['perplexity'], #default 30.0
            #early_exaggeration=['early_exaggeration'], #default 12.0
            #late_exaggeration=['late_exaggeration'], #default 1.0
            learning_rate=learning_rate, #default 200.0
            n_iter=n_epochs, #default 1000
            #n_iter_without_progress seems to not be used
            # min_grad_norm default 1e-07
            # metricstr ‘euclidean’ only (default ‘euclidean’)
            #initstr ‘random’ (default ‘random’)
            verbose=verbose,
            random_state=random_state,
            method = 'barnes_hut', #methodstr ‘barnes_hut’, ‘fft’ or ‘exact’ (default ‘barnes_hut’)
            #anglefloat (default 0.5)
            learning_rate_method = None, #learning_rate_methodstr ‘adaptive’, ‘none’ or None (default ‘adaptive’)
            n_neighbors=n_neighbors, #(default 90)
            #perplexity_max_iterint (default 100)
            #exaggeration_iterint (default 250)
            #pre_momentumfloat (default 0.5)
            #post_momentumfloat (default 0.8)
            #square_distancesboolean, default=True
            output_type='numpy'
        )

        self.n_neighbors = n_neighbors
        self.verbose = verbose

    def fit_transform(self, points):
        # if self.verbose:
        #     print('loading data ...')
        # start = time.time()
        # gdf = cudf.DataFrame()
        # for i in range(points.shape[1]):
        #     gdf['fea%d'%i] = points[:,i]
        # end = time.time()
        # self.opt_time = end -start

        # if self.verbose:
        #     print('Data load took {:.3f} seconds'.format(end - start))

        if self.verbose:
            print("making knn graph...")
        t0 = time.time()
        knn_cuml = cuNearestNeighbors(n_neighbors=self.n_neighbors)
        knn_cuml.fit(points)
        knn_graph_comp = knn_cuml.kneighbors_graph(points)
        t1 = time.time()
        if self.verbose:
            print("knn graph took", t1 - t0)

        if self.verbose:
            print('fitting...')
        start = time.time()
        projection = self.dr.fit_transform(points, knn_graph=knn_graph_comp)
        end = time.time()
        if self.verbose:
            print('CUML tsne took {:.3f} seconds'.format(end - start))

        self.opt_time = end - start

        del knn_cuml
        del knn_graph_comp
        # del gdf

        return projection
    

