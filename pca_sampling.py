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
            # Rather than doing nearest and farthest neighbors, just get the
            #   n_neighbors points at evenly spaced indices
            # So neighbors[i] could be x_i's closest, 10th closest, 20th closest, ... points
            # Start from 1 to avoid ||x_i - x_i||^2
            neighbors[i] = inds[1::int(n_points / n_neighbors)]
        else:
            # Get the nearest and furthest neighbors
            # inds[0] is just i since dist(x_i, x_i) = 0 is the min
            # So start from 1 for nearest neighbors
            neighbors[i] = np.concatenate([
                inds[1:1 + int(n_neighbors/2)], # nearest neighbors
                inds[-int(n_neighbors/2):] # distant neighbors
            ])

    return neighbors

def get_neighbor_x_dists(x, full_x_dists, n_neighbors, stride=False):
    # If optimizing by neighbors, only calculate high-dim distances for
    #   each point to its respective neighbors
    n_points = int(x.shape[0])
    x_dists = np.zeros([n_points, n_points])
    neighbors = get_neighbors(x, n_neighbors, stride=stride)
    for i in range(n_points):
        # FIXME -- made this change without checking it
        neighbor_dists = full_x_dists[i][neighbors[i]]

        x_dists[i][neighbors[i]] = neighbor_dists

def get_sampling_x_dists(n_samples, full_x_dists):
    n_points = int(full_x_dists.shape[0])
    sampling_rate = float(n_samples) / n_points
    mask = np.random.rand(n_points, n_points) < sampling_rate
    sparse_x_dists = full_x_dists * mask

    return sparse_x_dists, mask

def get_sampling_y_dists(n_samples, y_vecs):
    # y_i gets ||x_i - x_j||^2 force w.r.t. y_j if x_i and x_j are being sampled
    # So for each such force, also sample a y_k to calculate ||y_i - y_k||^2
    n_points = int(y_vecs.shape[0])
    sampling_rate = float(n_samples) / n_points
    mask = np.random.rand(n_points, n_points) < sampling_rate
    sparse_y_dists = np.sum(np.square(y_vecs), axis=-1) * mask

    return sparse_y_dists, mask

def get_dataset(desired_size, mnist=True):
    # Download datasets
    if not mnist:
        x, _ = make_swiss_roll(n_samples=desired_size, noise=0.001)
        labels = np.ones([int(x.shape[0])])
        return x, labels

    train, _ = tfk.datasets.mnist.load_data(path='mnist.npz')
    x, labels = train
    x = x.astype(np.float32)

    # Resize MNIST dataset to be desired number of points
    dataset_size = int(x.shape[0])
    downsample_stride = int(float(dataset_size) / desired_size)
    x, labels = x[::downsample_stride], labels[::downsample_stride]
    n_points = int(x.shape[0])
    x = np.reshape(x, [n_points, -1])
    x /= 255.0
    x = x.astype(np.float32)

    return x, labels, n_points

def sklearn_PCA(x, labels):
    # SKLearn library PCA implementation
    pca = PCA()
    y = pca.fit_transform(x)
    plt.scatter(y[:, 0], y[:, 1], c=labels)
    plt.show()
    # plt.savefig(os.path.join('images', 'True_PCA.png'))
    # plt.clf()

def zero_row_cols(dists, mask):
    # Zero row/col means
    dists -= np.sum(dists, axis=1, keepdims=True) / np.sum(mask, axis=1, keepdims=True)
    dists -= np.sum(dists, axis=0, keepdims=True) / np.sum(mask, axis=0, keepdims=True)
    dists *= mask
    return dists

def get_vecs(array):
    return np.expand_dims(array, 0) - np.expand_dims(array, 1)

def grad_descent_PCA(
    x,
    y,
    labels,
    n_points,
    optimize_by_neighbors,
    optimize_by_sampling,
    momentum=False,
    n_samples=50
):
    lr = 1.0
    forces = np.zeros_like(y)

    if not (optimize_by_neighbors or optimize_by_sampling):
        # If n_samples is set to n_points, then it's just regular PCA gradient descent
        n_samples = n_points

    x_vecs = get_vecs(x)
    full_x_dists = np.sum(
        np.square(x_vecs),
        axis=-1
    )
    if not optimize_by_sampling:
        # if optimizing by sampling, need to zero out row/col means every epoch
        if optimize_by_neighbors:
            x_dists = get_neighbor_x_dists(
                x,
                full_x_dists,
                n_samples,
                stride=False
            )
        else:
            x_dists = full_x_dists
            mask = np.ones_like(x_dists)
        x_dists = zero_row_cols(x_dists, mask)

    n_epochs = 1000
    for i_epoch in tqdm(range(n_epochs), total=n_epochs):
        if optimize_by_sampling:
            x_dists, mask = get_sampling_x_dists(n_samples, full_x_dists)
            x_dists = zero_row_cols(x_dists, mask)

        y_vecs = get_vecs(y)
        if optimize_by_neighbors or optimize_by_sampling:
            y_dists, mask = get_sampling_y_dists(n_samples, y_vecs)
        else:
            y_dists = np.sum(np.square(y_vecs), axis=-1)
            mask = np.ones_like(y_dists)

        # zero row/col mean y distances
        y_dists = zero_row_cols(y_dists, mask)

        # PCA kernels
        Y_forces = - 4 / (n_points * n_samples) * y_dists
        Y_grads = np.expand_dims(Y_forces, -1) * y_vecs
        Y_grads = np.sum(Y_grads, axis=0)
        X_forces = 4 / (n_points * n_samples) * x_dists
        X_grads = np.expand_dims(X_forces, -1) * y_vecs
        X_grads = np.sum(X_grads, axis=0)

        if momentum:
            forces = forces * 0.9 + lr * (Y_grads + X_grads)
        else:
            forces = lr * (Y_grads + X_grads)
        y += forces
        
        if i_epoch % 50 == 0:
            plt.scatter(y[:, 0], y[:, 1], c=labels)
            plt.show()
            # plt.savefig(os.path.join('images', '{0:04d}.png'.format(i_epoch)))
            plt.clf()

    plt.scatter(y[:, 0], y[:, 1], c=labels)
    plt.show()

if __name__ == '__main__':
    optimize_by_neighbors = False
    optimize_by_sampling = False
    momentum = True
    n_samples = 100
    assert not(optimize_by_neighbors and optimize_by_sampling)

    desired_size = 1000
    x, labels, n_points = get_dataset(desired_size, mnist=True)
    y = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_points)
    # sklearn_PCA(x, labels)
    grad_descent_PCA(
        x,
        y,
        labels,
        n_points,
        optimize_by_neighbors,
        optimize_by_sampling,
        momentum=momentum,
        n_samples=n_samples
    )
