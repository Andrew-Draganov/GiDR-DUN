import numpy as np
from tensorflow import keras as tfk
from umap_ import UMAP
import argparse
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--random-init', action='store_true', help='whether to initialize randomly')
parser.add_argument('--ignore-umap-metric', action='store_true', help='If true, do NOT subtract rho\'s in the umap pseudo-distance metric')
parser.add_argument('--downsample_stride', type=int, default=10)
args = parser.parse_args()

init = 'spectral'
if args.random_init:
    init = 'random'

train, _ = tfk.datasets.mnist.load_data(path='mnist.npz')
x_train, y_train = train
x_train, y_train = x_train[::args.downsample_stride], y_train[::args.downsample_stride]
num_samples = int(x_train.shape[0])
x_train = np.reshape(x_train, [num_samples, -1])

dr = UMAP(random_state=12345, init=init, pseudo_distance=(not args.ignore_umap_metric))
print('fitting...')
dr.fit(x_train)
projection = dr.transform(x_train)

plt.scatter(projection[:, 0], projection[:, 1], c=y_train, s=1)
plt.show()
