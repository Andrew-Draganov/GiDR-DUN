import os
import time
from tqdm import tqdm
import numpy as np
from umap_ import UniformUmap
from data_loader import get_dataset
from distances import make_dist_plots, remove_diag
import argparse
from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from matplotlib import pyplot as plt

# FIXME - Make script that, for a dataset, saves the images of
#   - original UMAP
#   - original tSNE
#   - my UMAP without umap-kernel, pseudo-distance, umap symmetrization
#   - my tSNE with umap-kernel, pseudo-distance, umap symmetrization
#   - my UMAP with normalization
#       - with/without momentum?
#   - my tSNE without normalization
#       - with/without momentum?
#   - my UMAP without symm-attraction
#   - my tSNE with sym-attraction
#   - my UMAP with random_init
#   - my tSNE with laplacian_init
# And saves relevant metrics!!!
dataset_names = ['mnist']
num_points = 6000

algorithms = {
    # UMAP(verbose=True),
    # TSNE(verbose=True),
    'UMAP_w_tsne_kern_sym_dist': UniformUmap(
        optimize_method='cy_umap_sampling',
        pseudo_distance=False,
        tsne_symmetrization=True,
        a=1,
        b=1,
        verbose=True
    ),
    'tSNE_w_umap_kern_sym_dist': UniformUmap(
        optimize_method='cy_barnes_hut',
        pseudo_distance=True,
        tsne_symmetrization=False,
        a=None,
        b=None,
        verbose=True
    )
}

for dataset_name in dataset_names:
    points, labels = get_dataset(dataset_name, num_points)
    for i, (alg_name, algorithm) in enumerate(algorithms.items()):
        print('fitting %d...' % i)
        print(alg_name)
        print(algorithm)
        quit()
        start = time.time()
        projection = dr.fit_transform(points)
        end = time.time()
        plt.scatter(projection[:, 0], projection[:, 1], c=labels, s=0.1, alpha=0.6)
        path = os.path.join('images', alg_name, dataset_name) + '.png'
        plt.savefig(path)
        plt.clf()

        print('Total time took {:.3f} seconds'.format(end - start))
