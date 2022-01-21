import os
from tqdm import tqdm
import numpy as np
from tensorflow import keras as tfk
from sklearn.decomposition import PCA
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

n_points = 1000
# x, _ = make_swiss_roll(n_samples=n_points, noise=0.001)
# labels = np.ones([int(x.shape[0])])
train, _ = tfk.datasets.mnist.load_data(path='mnist.npz')
x, labels = train
x = x.astype(np.float32)
y = np.random.multivariate_normal([0, 0], [[0.01, 0], [0, 0.01]], n_points)

num_samples = int(x.shape[0])
downsample_stride = int(float(num_samples) / n_points)
x, labels = x[::downsample_stride], labels[::downsample_stride]
num_samples = int(x.shape[0])
x = np.reshape(x, [num_samples, -1])
x /= 255.0

pca = PCA()
y = pca.fit_transform(x)
plt.scatter(y[:, 0], y[:, 1], c=labels)
plt.savefig(os.path.join('images', 'True_PCA.png'))
plt.clf()

y = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_points)

lr = 1.0
forces = np.zeros_like(y)
x_dists = np.sum(np.square(np.expand_dims(x, 0) - np.expand_dims(x, 1)), axis=-1)
x_dists -= np.mean(x_dists, axis=1, keepdims=True)
x_dists -= np.mean(x_dists, axis=0, keepdims=True)
n_epochs = 1000
for i_epoch in tqdm(range(n_epochs), total=n_epochs):
    y_vecs = np.expand_dims(y, 0) - np.expand_dims(y, 1)
    
    # zero row/col mean y distances
    y_dists = np.sum(np.square(y_vecs), axis=-1)
    y_dists -= np.mean(y_dists, axis=1, keepdims=True)
    y_dists -= np.mean(y_dists, axis=0, keepdims=True)

    attractive_forces = - 4 / (n_points**2) * y_dists
    attractive_grads = np.expand_dims(attractive_forces, -1) * y_vecs
    attractive_grads = np.sum(attractive_grads, axis=0)

    repulsive_forces = 4 / (n_points**2) * x_dists
    repulsive_grads = np.expand_dims(repulsive_forces, -1) * y_vecs
    repulsive_grads = np.sum(repulsive_grads, axis=0)

    # momentum gradient descent
    forces = forces * 0.9 + lr * (attractive_grads + repulsive_grads)
    y += forces
    
    if i_epoch % 100 == 0:
        plt.scatter(y[:, 0], y[:, 1], c=labels)
        plt.savefig(os.path.join('images', '{0:04d}.png'.format(i_epoch)))
        plt.clf()
plt.scatter(y[:, 0], y[:, 1], c=labels)
plt.show()
