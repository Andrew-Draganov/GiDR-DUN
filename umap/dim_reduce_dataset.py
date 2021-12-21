import time
from tqdm import tqdm
import numpy as np
from tensorflow import keras as tfk
import tensorflow_datasets as tfds
from umap_ import UniformUmap
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
    '--normalization',
    choices=['tsne', 'umap'],
    default='umap',
    help='Whether to use UMAP or tSNE normalization'
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
if args.dataset == 'mnist':
    train, _ = tfk.datasets.mnist.load_data(path='mnist.npz')
    points, labels = train
elif args.dataset == 'cifar':
    train, _ = tfk.datasets.cifar10.load_data()
    points, labels = train
elif args.dataset == 'swiss_roll':
    num_samples = 50000
    points, _ = make_swiss_roll(n_samples=num_samples, noise=0.01)
    labels = np.arange(num_samples)
elif args.dataset == 'google_news':
    import os, csv
    # To run this dataset, download https://data.world/jaredfern/googlenews-reduced-200-d
    #   and place it into the directory 'datasets'
    file = open(os.path.join('datasets', 'gnews_mod.csv'), 'r')
    reader = csv.reader(file)
    num_points = min(args.num_points, 350000)
    points = np.zeros([num_points, 200])
    for i, line in tqdm(enumerate(reader), total=num_points):
        # First line is column descriptions
        if i == 0:
            continue
        if i > num_points:
            break
        for j, element in enumerate(line[1:]): # First column is string text
            points[i-1, j] = float(element)
    labels = np.ones([num_points])
elif args.dataset == 'coil':
    num_samples = 7200
    dataset, _ = tfds.load(
        'coil100',
        split=['train'],
        with_info=True
    )
    dataset = dataset[0].batch(1)
    points = np.zeros([num_samples, 128, 128, 3])
    labels = np.zeros([num_samples])
    for i, element in enumerate(dataset):
        points[i] = element['image']
        labels[i] = element['angle_label']
else:
    raise ValueError("Unsupported dataset")

num_samples = int(points.shape[0])
if args.num_points > num_samples:
    print('WARNING: requested %d samples but dataset only has %d elements' % \
          (args.num_points, num_samples))
downsample_stride = int(float(num_samples) / args.num_points)
points, labels = points[::downsample_stride], labels[::downsample_stride]
num_samples = int(points.shape[0])
points = np.reshape(points, [num_samples, -1])

# FIXME - do I need this normalization?
points = np.array(points) / 255.0

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
            normalization=args.normalization,
            sym_attraction=args.sym_attraction,
            euclidean=not args.angular,
            momentum=args.momentum,
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

# count = 0
# base_Z = 0
# scaled_Z = 0
# scalar = 3
# laurent = 0
# taylor = 0
# for i, x in tqdm(enumerate(points), total=args.num_points):
#     for j, y in enumerate(points):
#         if i != j:
#             dist = np.sum(np.square(x - y))
#             if dist < 2:
#                 count += 1
#             base_Z += 1 / (1 + dist)
#             scaled_Z += 1 / (1 + scalar**2 * dist)
#             laurent += 1 / (scalar * dist) - 1 / (scalar**2 * dist**2) + 1/(scalar**3 * dist**3)
#             taylor += 1 - (scalar * dist) + (scalar**2 * dist**2) - (scalar**3 * dist**3)
# 
# print('Original Z is {}'.format(base_Z))
# print('After scaling the points by {}, the new Z is {}'.format(scalar, scaled_Z))
# print('The Laurent series approximation for the scaled Z is {}'.format(laurent))
# print('The Taylor series approximation for the scaled Z is {}'.format(taylor))
# print('Only {} pairs of points have distance < 2'.format(count))

plt.scatter(projection[:, 0], projection[:, 1], c=labels, s=0.1, alpha=0.8)
plt.show()
