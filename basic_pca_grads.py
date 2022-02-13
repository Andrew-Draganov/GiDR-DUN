import os
from tqdm import tqdm
import numpy as np
from tensorflow import keras as tfk
from sklearn.decomposition import PCA
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

def get_dataset(desired_size, mnist=True):
    # Download datasets
    if not mnist:
        x, _ = make_swiss_roll(n_samples=desired_size, noise=0.001)
        labels = np.ones([int(x.shape[0])])
        return x, labels, desired_size

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

def zero_row_cols(dists):
    # Zero row/col means
    dists -= np.mean(dists, axis=1, keepdims=True)
    dists -= np.mean(dists, axis=0, keepdims=True)

def get_vecs(array):
    return np.expand_dims(array, 0) - np.expand_dims(array, 1)

def dist_based_PCA(
    x,
    y,
    labels,
    n_points,
    momentum=False,
):
    lr = 1.0
    forces = np.zeros_like(y)

    x_vecs = get_vecs(x)
    x_dists = np.sum(np.square(x_vecs), axis=-1)
    zero_row_cols(x_dists)

    n_epochs = 500
    for i_epoch in tqdm(range(n_epochs), total=n_epochs):
        y_vecs = get_vecs(y)
        y_dists = np.sum(np.square(y_vecs), axis=-1)

        # zero row/col mean y distances
        zero_row_cols(y_dists)

        # PCA kernels
        Y_forces = - 4 / (n_points * n_points) * y_dists
        Y_grads = np.expand_dims(Y_forces, -1) * y_vecs
        Y_grads = np.sum(Y_grads, axis=0)

        X_forces = 4 / (n_points * n_points) * x_dists
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

def plot_gram_approx(x, y, n_points):
    G_x = np.sum(np.expand_dims(x, 0) * np.expand_dims(x, 1), axis=-1)
    G_y = np.sum(np.expand_dims(y, 0) * np.expand_dims(y, 1), axis=-1)
    LG = (G_y - G_x) - np.mean(G_y - G_x, axis=0, keepdims=True)
    full_grads = np.matmul(LG, y)
    grad_alignments = np.zeros([n_points-1])

    for i in tqdm(range(1, n_points), total=n_points-1):
        y_mask = np.ones_like(y)
        y_samples = np.random.choice(np.arange(n_points), size=[n_points, 1], replace=False)
        y_mask *= y_samples < i
        i_y = y * y_mask

        iG_y = np.sum(np.expand_dims(i_y, 0) * np.expand_dims(i_y, 1), axis=-1)
        iLG = (iG_y - G_x) - np.mean(iG_y - G_x, axis=0, keepdims=True)
        i_grads = np.matmul(iLG, y)

        grad_alignment = np.sum(i_grads * full_grads) / np.sum(full_grads * full_grads)
        grad_alignments[i-1] = grad_alignment

    plt.scatter(np.arange(1, n_points), grad_alignments)
    plt.title('PCA gradient estimation as a function of # samples')
    plt.ylabel('dot(estimated_y_grad, total_y_grad) / ||total_y_grad||')
    plt.xlabel('Number of Y samples taken')
    plt.show()


def gram_mat_PCA(
    x,
    y,
    labels,
    n_points,
    momentum=False,
):
    lr = 0.0001
    forces = np.zeros_like(y)
    G_x = np.sum(np.expand_dims(x, 0) * np.expand_dims(x, 1), axis=-1)

    n_epochs = 500
    for i_epoch in tqdm(range(n_epochs), total=n_epochs):
        G_y = np.sum(np.expand_dims(y, 0) * np.expand_dims(y, 1), axis=-1)
        LG = (G_y - G_x) - np.mean(G_y - G_x, axis=0, keepdims=True)
        grads = np.matmul(LG, y)

        if momentum:
            forces = forces * 0.9 + lr * grads
        else:
            forces = lr * grads
        y -= forces
        
        if i_epoch % 50 == 0:
            plt.scatter(y[:, 0], y[:, 1], c=labels)
            plt.show()
            # plt.savefig(os.path.join('images', '{0:04d}.png'.format(i_epoch)))
            plt.clf()

    plt.scatter(y[:, 0], y[:, 1], c=labels)
    plt.show()

if __name__ == '__main__':
    momentum = False
    desired_size = 1000
    x, labels, n_points = get_dataset(desired_size, mnist=True)
    y = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_points)
    # sklearn_PCA(x, labels)
    # grad_descent_PCA(
    #     x,
    #     y,
    #     labels,
    #     n_points,
    #     momentum=momentum,
    # )
    plot_gram_approx(x, y, n_points)
    gram_mat_PCA(
        x,
        y,
        labels,
        n_points,
        momentum=momentum,
    )
