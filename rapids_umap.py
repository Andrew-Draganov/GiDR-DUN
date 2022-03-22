import time
import os
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
import pandas as pd
import numpy as np
from tensorflow import keras as tfk
#For printing
from uniform_umap.distances import make_dist_plots
from matplotlib import pyplot as plt
# GPU UMAP
import cudf
from cuml.manifold.umap import UMAP as cumlUMAP

train, _ = tfk.datasets.mnist.load_data(path='mnist.npz')
points, labels = train

print(points.shape)
t_points = points.reshape(points.shape[0], -1)
points = np.array(t_points) / 255.0
target = np.array([labels])

print(points.shape)

print('loading data ...')
start = time.time()
gdf = cudf.DataFrame()
for i in range(points.shape[1]):
    gdf['fea%d'%i] = points[:,i]
end = time.time()
print('Data load took {:.3f} seconds'.format(end - start))

print("making knn graph...")
t0 = time.time()
knn_cuml = cuNearestNeighbors(n_neighbors=15)
knn_cuml.fit(gdf)
knn_graph_comp = knn_cuml.kneighbors_graph(gdf)
t1 = time.time()
print("knn graph took", t1 - t0)

print('fitting...')
start = time.time()
projection = cumlUMAP(n_neighbors=15, n_epochs=500, negative_sample_rate=1, init="spectral").fit_transform(gdf, knn_graph=knn_graph_comp)
end = time.time()
print('CUML UMAP took {:.3f} seconds'.format(end - start))

make_dist_plots(points, projection, labels, "cumlumap", "mnist")

plt.scatter(projection[:, 0], projection[:, 1], c=labels, s=0.1, alpha=0.8)
plt.show()
