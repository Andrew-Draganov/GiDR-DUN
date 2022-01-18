import time
from tqdm import tqdm
import numpy as np
from umap_ import UniformUmap
from data_loader import get_dataset
from distances import make_dist_plots, remove_diag
import argparse
from sklearn.manifold import TSNE
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset',
    choices=['mnist', 'cifar', 'swiss_roll', 'coil', 'google_news'],
    default='mnist',
    help='Which dataset to apply algorithm to'
)
parser.add_argument(
    '--dr-algorithm',
    choices=['uniform_umap', 'original_umap', 'tsne', 'pca', 'kernel_pca'],
    default='uniform_umap',
    help='Which algorithm to use for performing dim reduction'
)
parser.add_argument(
    '--initialization',
    choices=['spectral', 'random'],
    default='spectral',
    help='Controls the method for initializing the low-dim representation'
)
parser.add_argument(
    '--tsne-symmetrization',
    action='store_true',
    help='When present, symmetrize using the tSNE method'
)
parser.add_argument(
    '--make-plots',
    action='store_true',
    help='When present, make plots regarding distance relationships. ' \
         'Requires a high downsample_stride to not run out of memory'
)

parser.add_argument(
    '--ignore-umap-metric',
    action='store_true',
    help='If true, do NOT subtract rho\'s in the umap pseudo-distance metric'
)
parser.add_argument(
    '--optimize-method',
    choices=[
        'torch_sgd',
        'torch',
        'umap_sampling',
        'umap_uniform',
        'cy_barnes_hut',
        'cy_umap_uniform',
        'cy_umap_sampling',
    ],
    default='cy_umap_uniform',
    help='Which optimization algorithm to use'
)
parser.add_argument(
    '--normalized',
    action='store_true',
    help='If true, normalize high- and low-dimensional pairwise likelihood matrices'
)
parser.add_argument(
    '--sym-attraction',
    action='store_true',
    help='Whether to attract along both ends of a nearest neighbor edge'
)
parser.add_argument(
    '--angular',
    action='store_true',
    help='When present, use cosine similarity metric on high-dimensional points'
)
parser.add_argument(
    '--momentum',
    action='store_true',
    help='Whether to perform mometnum gradient descent'
)
parser.add_argument(
    '--kernel-choice',
    choices=['tsne', 'umap'],
    default='umap',
    help='Which high- and low-dimensional kernels to use'
)
parser.add_argument(
    '--num-points',
    type=int,
    default=5000,
    help='Number of samples to use from the dataset'
)
parser.add_argument(
    '--batch-size',
    type=int,
    default=-1,
    help='Batch size for Uniform-UMAP optimization.'
         'If -1, defaults to regular gradient descent'
)
parser.add_argument(
    '--n-neighbors',
    type=int,
    default=15
)
parser.add_argument(
    '--neg-sample-rate',
    type=int,
    default=15,
    help='How many negative samples to use for each positive sample (in UMAP)'
)
parser.add_argument(
    '--n-epochs',
    type=int,
    default=500,
    help='Number of times to cycle through the dataset'
)
args = parser.parse_args()

print('Loading %s dataset...' % args.dataset)
points, labels = get_dataset(args.dataset, args.num_points)

if args.kernel_choice == 'tsne':
    a, b = 1, 1
else:
    assert args.kernel_choice == 'umap'
    a, b = None, None

if args.dr_algorithm == 'uniform_umap':
    # FIXME - change my umap code to have a different name/constructor
    dr = UniformUmap(
            n_neighbors=args.n_neighbors,
            n_epochs=args.n_epochs,
            random_state=12345, # Comment this out to turn on parallelization
            init=args.initialization,
            pseudo_distance=(not args.ignore_umap_metric),
            tsne_symmetrization=args.tsne_symmetrization,
            optimize_method=args.optimize_method,
            negative_sample_rate=args.neg_sample_rate,
            normalized=int(args.normalized),
            sym_attraction=args.sym_attraction,
            euclidean=not args.angular,
            momentum=args.momentum,
            batch_size=args.batch_size,
            a=a,
            b=b,
            verbose=True
        )
elif args.dr_algorithm == 'original_umap':
    from umap import UMAP
    dr = UMAP(
            n_neighbors=args.n_neighbors,
            n_epochs=args.n_epochs,
            # random_state=12345, # Comment this out to turn on parallelization
            init=args.initialization,
            negative_sample_rate=args.neg_sample_rate,
            a=a,
            b=b,
            verbose=True
        )
elif args.dr_algorithm == 'tsne':
    dr = TSNE(random_state=12345, verbose=3)
elif args.dr_algorithm == 'pca':
    dr = PCA()
elif args.dr_algorithm == 'kernel_pca':
    dr = KernelPCA(n_components=2)
else:
    raise ValueError("Unsupported algorithm")

print('fitting...')
start = time.time()
projection = dr.fit_transform(points)
end = time.time()
print('Total time took {:.3f} seconds'.format(end - start))
if args.make_plots:
    make_dist_plots(points, projection, labels, args.dr_algorithm, args.dataset)

plt.scatter(projection[:, 0], projection[:, 1], c=labels, s=0.1, alpha=0.6)
plt.show()
