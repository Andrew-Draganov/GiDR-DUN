import numpy as np
from tensorflow import keras as tfk
from umap_ import UMAP
from distances import make_dist_plots, remove_diag
# from umap import UMAP
import argparse
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
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
    default='umap_sampling',
    help='Which optimization algorithm to use'
)
# FIXME - rename to "normalize_P"
parser.add_argument(
    '--normalization',
    choices=['tsne', 'umap'],
    default='umap',
    help='Whether to use UMAP or tSNE normalization'
)
parser.add_argument(
    '--kernel-choice',
    choices=['tsne', 'umap'],
    default='umap',
    help='Which high- and low-dimensional kernels to use'
)
parser.add_argument(
    '--downsample-stride',
    type=int,
    default=15,
    help='Take every n-th sample from the dataset. Higher value -> smaller dataset'
)
parser.add_argument(
    '--dr-algorithm',
    choices=['umap', 'tsne', 'pca'],
    default='umap',
    help='Which algorithm to use for performing dim reduction'
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

print('Loading MNIST data...')
train, _ = tfk.datasets.mnist.load_data(path='mnist.npz')
# train, _ = tfk.datasets.cifar10.load_data()
x_train, y_train = train
x_train, y_train = x_train[::args.downsample_stride], y_train[::args.downsample_stride]
num_samples = int(x_train.shape[0])
x_train = np.reshape(x_train, [num_samples, -1])

### Expected value of distances
x_train = np.array(x_train) / 255.0

if args.kernel_choice == 'tsne':
    a, b = 1, 1
else:
    assert args.kernel_choice == 'umap'
    a, b = None, None

if args.dr_algorithm == 'umap':
    dr = UMAP(
            n_neighbors=args.n_neighbors,
            n_epochs=args.n_epochs,
            random_state=12345, # Comment this out to turn on parallelization
            init=args.initialization,
            pseudo_distance=(not args.ignore_umap_metric),
            tsne_symmetrization=args.tsne_symmetrization,
            optimize_method=args.optimize_method,
            negative_sample_rate=args.neg_sample_rate,
            normalization=args.normalization,
            kernel_choice=args.kernel_choice,
            a=a,
            b=b,
            verbose=True
        )
elif args.dr_algorithm == 'tsne':
    dr = TSNE(random_state=12345)
else:
    dr = PCA()

print('fitting...')
projection = dr.fit_transform(x_train)
if args.make_plots:
    make_dist_plots(x_train, projection, y_train, args.dr_algorithm)

plt.scatter(projection[:, 0], projection[:, 1], c=y_train, s=1)
plt.show()
