import numpy as np
from tensorflow import keras as tfk
from umap_ import UMAP
from distances import make_dist_plots
# from umap import UMAP
import argparse
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    '--random-init',
    action='store_true',
    help='whether to initialize randomly'
)
parser.add_argument(
    '--tsne-symmetrization',
    action='store_true',
    help='whether to symmetrize akin to tSNE method'
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
        'barnes_hut'
    ],
    default='umap_sampling',
    help='Which optimization algorithm to use'
)
# FIXME - rename to "normalize_P"
parser.add_argument(
    '--normalization',
    choices=['tsne', 'umap'],
    default='umap',
    help='Which optimization algorithm to use'
)
parser.add_argument(
    '--kernel-choice',
    choices=['tsne', 'umap'],
    default='umap',
    help='Which weight normalization and scaling to use'
)
parser.add_argument(
    '--downsample-stride',
    type=int,
    default=15
)
parser.add_argument(
    '--dr-algorithm',
    choices=['umap', 'tsne', 'pca'],
    default='umap',
    help='Which algorithm to use to save images'
)
parser.add_argument(
    '--n-neighbors',
    type=int,
    default=15
)
parser.add_argument(
    '--neg-sample-rate',
    type=int,
    default=15
)
parser.add_argument(
    '--n-epochs',
    type=int,
    default=500
)
args = parser.parse_args()

init = 'spectral'
if args.random_init:
    init = 'random'

def remove_diag(A):
    return A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],-1)

print('Loading MNIST data...')
train, _ = tfk.datasets.mnist.load_data(path='mnist.npz')
# train, _ = tfk.datasets.cifar10.load_data()
x_train, y_train = train
x_train, y_train = x_train[::args.downsample_stride], y_train[::args.downsample_stride]
num_samples = int(x_train.shape[0])
x_train = np.reshape(x_train, [num_samples, -1])

### Expected value of distances
x_train = np.array(x_train) / 255.0

# cov = np.zeros([784, 784])
# np.fill_diagonal(cov, np.var(x_train, axis=0))
# print('sampling...')
# x_train = np.random.multivariate_normal(
#     0 * np.mean(x_train, axis=0), 
#     cov,
#     num_samples
# )
# all_dists = np.expand_dims(x_train, 0) - np.expand_dims(x_train, 1)
# all_dists = np.sum(np.square(all_dists), -1)
# print(all_dists.shape)
# mean_dist = np.mean(all_dists)
# V = np.var(x_train, axis=0)
# expected_dist = 2 * np.sum(V)
# print(mean_dist)
# print(expected_dist)
# dist_var = np.var(all_dists)
# expected_var = np.sum(4 * V ** 4 + 4 * V ** 2)
# print(dist_var)
# print(expected_var)
# quit()

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
            init=init,
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
make_dist_plots(x_train, projection, y_train, args.dr_algorithm)

# plt.scatter(projection[:, 0], projection[:, 1], c=y_train, s=1)
# plt.show()
