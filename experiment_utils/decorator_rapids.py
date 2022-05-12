from cuml.manifold.umap import UMAP as cumlUMAP
from cuml.manifold import TSNE as cumlTSNE
import time
import os

class rapids_umap:
    def __init__(
        self,
        n_neighbors,
        n_components,
        n_epochs,
        negative_sample_rate,
        a,
        b,
        random_state,
        verbose=True
    ):
        self.dr = cumlUMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            n_epochs=n_epochs,
            negative_sample_rate=negative_sample_rate,
            a=a,
            b=b,
            random_state=random_state,
            verbose=verbose,
            output_type='numpy'
        )

        self.n_neighbors = n_neighbors
        self.verbose = verbose
    
    def fit_transform(self, points):
        if self.verbose:
            print('fitting...')
        start = time.time()
        projection = self.dr.fit_transform(points)
        end = time.time()
        if self.verbose:
            print('CUML UMAP took {:.3f} seconds'.format(end - start))
        self.opt_time = end - start
        return projection

class rapids_tsne:
    def __init__(
        self,
        n_components,
        n_epochs,
        random_state,
        n_neighbors,
        verbose=True
    ):
        self.dr = cumlTSNE(
            n_components=n_components,
            n_iter=n_epochs,
            verbose=verbose,
            random_state=random_state,
            method = 'barnes_hut',
            learning_rate_method=None,
            n_neighbors=n_neighbors,
            output_type='numpy'
        )

        self.n_neighbors = n_neighbors
        self.verbose = verbose

    def fit_transform(self, points):
        if self.verbose:
            print("making knn graph...")
            print('fitting...')
        start = time.time()
        projection = self.dr.fit_transform(points)
        end = time.time()
        if self.verbose:
            print('CUML tsne took {:.3f} seconds'.format(end - start))
        self.opt_time = end - start
        return projection
