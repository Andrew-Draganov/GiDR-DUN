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
def save_image(alg_name, dataset_name):
    dir_path = os.path.join('images', alg_name)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, dataset_name + '.png')
    plt.savefig(file_path)
    plt.clf()

dataset_names = ['mnist']
num_points = 60000

algorithms = {
    'base_umap': UMAP(verbose=True),
    'base_tsne': TSNE(verbose=True),
    'uniform_umap': UniformUmap(
        optimize_method='cy_umap_uniform',
        sym_attraction=True
    ),
    'uniform_umap_doing_tSNE': UniformUmap(
        optimize_method='cy_umap_uniform',
        normalized=True,
        momentum=True
    ),
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
        init='random',
        momentum=True,
        normalized=True,
        sym_attraction=False,
        verbose=True
    ),
    'UMAP_w_normalization_and_momentum': UniformUmap(
        optimize_method='cy_umap_uniform',
        normalized=True,
        momentum=True,
        verbose=True
    ),
    'tSNE_w_sym_attr_without_norm_and_momentum': UniformUmap(
        optimize_method='cy_barnes_hut',
        init='random',
        sym_attraction=True,
        verbose=True
    ),
    'UMAP_without_sym_attr': UniformUmap(
        optimize_method='cy_umap_sampling',
        sym_attraction=False,
        verbose=True
    ),
    'tSNE_w_sym_attr': UniformUmap(
        optimize_method='cy_barnes_hut',
        init='random',
        normalized=True,
        momentum=True,
        verbose=True
    ),
    'UMAP_w_rand_init': UniformUmap(
        optimize_method='cy_umap_sampling',
        init='random',
        verbose=True
    ),
    'tSNE_w_laplace_init': UniformUmap(
        optimize_method='cy_barnes_hut',
        normalized=True,
        momentum=True,
        sym_attraction=False,
        verbose=True
    ),
}

for dataset_name in dataset_names:
    for i, (alg_name, algorithm) in enumerate(algorithms.items()):
        print('\nfitting %d...' % i)
        points, labels = get_dataset(dataset_name, num_points)
        start = time.time()
        projection = algorithm.fit_transform(points)
        end = time.time()
        plt.scatter(projection[:, 0], projection[:, 1], c=labels, s=0.1, alpha=0.6)
        save_image(alg_name, dataset_name)
        print('Total time took {:.3f} seconds'.format(end - start))
