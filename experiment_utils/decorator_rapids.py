import time
import os
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors

class rapids_wrapper:
    def __init__(
        self,
        n_neighbors,
        n_components,
        random_state,
        negative_sample_rate=5,
        n_epochs=500,
        a=None,
        b=None,
        umap=True,
        barnes_hut=True,
        verbose=True
    ):
        if umap:
            from cuml.manifold.umap import UMAP as cumlUMAP
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
        else:
            from cuml.manifold import TSNE as cumlTSNE
            self.dr = cumlTSNE(
                n_components=n_components,
                verbose=verbose,
                n_iter=n_epochs,
                random_state=random_state,
                method='barnes_hut' if barnes_hut else 'fft', # FIXME -- do both in experiments??
                learning_rate_method=None,
                n_neighbors=n_neighbors,
                output_type='numpy'
            )

        self.n_neighbors = n_neighbors
        self.verbose = verbose

    def fit_transform(self, points):
        if self.verbose:
            print("making knn graph...")
        # Have to make the KNN graph separately so that we can free resources at the end
        # of this method
        knn_start = time.time()
        knn_cuml = cuNearestNeighbors(n_neighbors=self.n_neighbors, init="random")
        knn_cuml.fit(points)
        knn_graph_comp = knn_cuml.kneighbors_graph(points)
        knn_end = time.time()
        if self.verbose:
            print("knn graph took", knn_end - knn_start)

        if self.verbose:
            print('fitting...')
        embed_start = time.time()
        projection = self.dr.fit_transform(points, knn_graph=knn_graph_comp)
        embed_end = time.time()
        if self.verbose:
            print('CUML UMAP took {:.3f} seconds'.format(embed_end - embed_start))

        self.opt_time = embed_end - embed_start
        self.total_time = embed_end - knn_start

        # RAPIDS does not clean up GPU resources fully
        #   Therefore we have to do this or else multiple experiments back-to-back
        #   will run out of memory
        del knn_cuml
        del knn_graph_comp
        return projection
