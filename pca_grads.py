import os
from tqdm import tqdm
import numpy as np
from tensorflow import keras as tfk
from sklearn.decomposition import PCA
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

def get_neighbors(x, n_neighbors=20, stride=False):
    n_points = int(x.shape[0])
    neighbors = {i: None for i in range(n_points)}

    for i, x_i in tqdm(enumerate(x), total=n_points):
        sq_dists = np.sum(np.square(np.expand_dims(x_i, 0) - x), -1)
        sort_indices = np.argsort(sq_dists)
        inds = np.arange(n_points)[sort_indices]

        if stride:
            neighbors[i] = inds[::int(n_points / n_neighbors)]
        else:
            # Get the nearest and furthest neighbors
            # inds[0] is just i since dist(x_i, x_i) = 0 is the min
            # So start from 1 for nearest neighbors
            neighbors[i] = np.concatenate([
                inds[1:1 + int(n_neighbors/2)], # nearest neighbors
                inds[-int(n_neighbors/2):] # distant neighbors
            ])

    return neighbors


optimize_by_neighbors = True
desired_size = 1000
# x, _ = make_swiss_roll(n_samples=n_points, noise=0.001)
# labels = np.ones([int(x.shape[0])])

train, _ = tfk.datasets.mnist.load_data(path='mnist.npz')
x, labels = train
x = x.astype(np.float32)

dataset_size = int(x.shape[0])
downsample_stride = int(float(dataset_size) / desired_size)
x, labels = x[::downsample_stride], labels[::downsample_stride]
n_points = int(x.shape[0])
x = np.reshape(x, [n_points, -1])
x /= 255.0
x = x.astype(np.float32)

# pca = PCA()
# y = pca.fit_transform(x)
# plt.scatter(y[:, 0], y[:, 1], c=labels)
# plt.savefig(os.path.join('images', 'True_PCA.png'))
# plt.clf()

y = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_points)

# FIXME FIXME FIXME -- what happens if we only do this with nearest/farthest neighbors???
lr = 1.0
forces = np.zeros_like(y)
X_forces = np.zeros_like(y)
Y_forces = np.zeros_like(y)
if optimize_by_neighbors:
    n_neighbors = 50
    neighbors = get_neighbors(x, n_neighbors, stride=False)
else:
    n_neighbors = n_points

x_dists = np.zeros([n_points, n_points])
y_dists = np.zeros([n_points, n_points])

if optimize_by_neighbors:
    for i in range(n_points):
        neighbor_vecs = np.expand_dims(x[i], 0) - x[neighbors[i]]
        neighbor_dists = np.sum(np.square(neighbor_vecs), axis=-1)
        x_dists[i][neighbors[i]] = neighbor_dists
else:
    x_dists = np.sum(np.square(np.expand_dims(x, 0) - np.expand_dims(x, 1)), axis=-1)

# Zero row/col means
x_dists -= np.mean(x_dists, axis=1, keepdims=True)
x_dists -= np.mean(x_dists, axis=0, keepdims=True)
n_epochs = 1000
for i_epoch in tqdm(range(n_epochs), total=n_epochs):
    y_vecs = np.expand_dims(y, 0) - np.expand_dims(y, 1)

    if optimize_by_neighbors:
        n_samples = n_neighbors
        random_row_inds = {
            i: np.random.choice(np.arange(n_points), size=n_samples, replace=False)
            for i in range(n_points)
        }
        random_col_inds = {
            i: np.random.choice(np.arange(n_points), size=n_samples, replace=False)
            for i in range(n_points)
        }
        y_dists *= 0
        for i in range(n_points):
            sampled_row_dists = np.sum(np.square(y_vecs[i][random_row_inds[i]]), -1)
            y_dists[i][random_row_inds[i]] = sampled_row_dists
            sampled_col_dists = np.sum(np.square(y_vecs[:, i][random_col_inds[i]]), -1)
            y_dists[:, i][random_col_inds[i]] = sampled_col_dists
    else:
        y_dists = np.sum(np.square(y_vecs), axis=-1)

    # zero row/col mean y distances
    y_dists -= np.mean(y_dists, axis=1, keepdims=True)
    y_dists -= np.mean(y_dists, axis=0, keepdims=True)

    # PCA kernels
    Y_forces = - 4 / (n_points * n_neighbors) * y_dists
    Y_grads = np.expand_dims(Y_forces, -1) * y_vecs
    Y_grads = np.sum(Y_grads, axis=0)

    X_forces = 4 / (n_points * n_neighbors) * x_dists
    X_grads = np.expand_dims(X_forces, -1) * y_vecs
    X_grads = np.sum(X_grads, axis=0)

    # momentum gradient descent
    # forces = forces * 0.9 + lr * (Y_grads + X_grads)
    forces = lr * (Y_grads + X_grads)
    y += forces
    
    if i_epoch % 50 == 0:
        plt.scatter(y[:, 0], y[:, 1], c=labels)
        plt.show()
        # plt.savefig(os.path.join('images', '{0:04d}.png'.format(i_epoch)))
        plt.clf()
plt.scatter(y[:, 0], y[:, 1], c=labels)
plt.show()
