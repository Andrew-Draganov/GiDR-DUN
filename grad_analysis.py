import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
from tensorflow import keras as tfk

# Low- and High-Dim distance getters
def get_simple_dists(upper_bound):
    low_dim_dists = np.arange(0.01, upper_bound, 0.01)
    high_dim_dists = np.arange(0.01, upper_bound, 0.01)
    return low_dim_dists, high_dim_dists

def get_exp_dists():
    low_dim_dists = [0.005 * 1.12 ** i for i in range(100)]
    high_dim_dists = [0.005 * 1.12 ** i for i in range(100)]
    return low_dim_dists, high_dim_dists

def get_pairwise_dists(points):
    dists = []
    for i, x_i in enumerate(points):
        for x_j in points[i:]:
            if not np.all(x_i == x_j):
                dists.append(np.sum(np.square(x_i - x_j)))
    return np.sort(dists)

def get_random_low_dim_dists(low_dimensionality, num_points):
    low_dim_points = np.random.multivariate_normal(
        np.zeros(low_dimensionality),
        np.eye(low_dimensionality),
        size=num_points
    )
    return get_pairwise_dists(low_dim_points)

def get_gaussian_dists(high_dimensionality, low_dimensionality, num_points):
    high_dim_points = np.random.multivariate_normal(
        np.zeros(high_dimensionality),
        np.eye(high_dimensionality),
        size=num_points
    )
    high_dim_dists = get_pairwise_dists(high_dim_points)
    low_dim_dists = get_random_low_dim_dists(low_dimensionality, num_points)

    return low_dim_dists, high_dim_dists

def get_mnist_dists(num_points):
    train, _ = tfk.datasets.mnist.load_data(path='mnist.npz')
    points, labels = train
    num_samples = int(points.shape[0])
    if num_points > num_samples:
        print('WARNING: requested %d samples but dataset only has %d elements' % \
              (args.num_points, num_samples))
    downsample_stride = int(float(num_samples) / num_points)
    points, labels = points[::downsample_stride], labels[::downsample_stride]
    num_samples = int(points.shape[0])
    points = np.reshape(points, [num_samples, -1]).astype(np.float32)
    high_dim_dists = get_pairwise_dists(points)
    low_dim_dists = get_random_low_dim_dists(2, num_points)

    return low_dim_dists, high_dim_dists



# GRADIENT DERIVATIONS
def tsne_grads(Dx, Dy):
    P = np.exp(-(np.square(Dx) / 2))
    Q = 1 / (1 + np.square(Dy))
    P, Q = np.meshgrid(P, Q)
    Z_stack = [1 / (1 + np.square(Dy)) for _ in Dy]
    Dy_stack = [Dy for _ in Dy]
    gradient = 4 * (P - Q) * np.stack(Z_stack, -1) * np.stack(Dy_stack, -1)
    gradient = np.flip(gradient, 0)
    return gradient

def umap_grads(Dx, Dy, a, b, k):
    sigma = find_sigma(Dx, np.log2(k))
    P_vec = np.exp(-1 * (np.square(Dx) - np.min(np.square(Dx))) / sigma)
    attr_vec = -2 * a * b * np.power(np.square(Dy), b - 1) / (1 + np.square(Dy))
    rep_vec = 2 * b / ((0.001 + np.square(Dy)) * (1 + a * np.power(np.square(Dy), b)))

    P, attr = np.meshgrid(P_vec, attr_vec)
    _, rep = np.meshgrid(P_vec, rep_vec)
    Dy_stack = [Dy for _ in Dy]
    gradient = (P * attr + (1 - P) * rep) * np.stack(Dy_stack, -1)
    gradient = -1 * np.flip(gradient, 0)
    return gradient

# UTILS
def find_sigma(D, target):
    error = 1
    lo = 0.0
    hi = 100000
    mid = 1.0
    for i in range(100):
        P_vec = np.exp(-1 * (np.square(D) - np.min(np.square(D))) / mid)
        psum = np.sum(P_vec)
        error = np.abs(psum - target)
        if np.fabs(psum - target) < 0.01:
            break
        if psum > target:
            hi = mid
            mid = (lo + hi) / 2.0
        else:
            lo = mid
            if hi == 100000:
                mid *= 2
            else:
                mid = (lo + hi) / 2.0
    return mid


def make_plot(num_dists, gradient):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    X, Y = np.meshgrid(np.arange(num_dists), np.arange(num_dists))
    ax1.matshow(gradient, cmap=plt.get_cmap('viridis'))
    PC = ax1.pcolor(X, Y, gradient)
    contour = ax1.contour(
        X,
        Y,
        gradient,
        levels=[0, 10],
        colors='black',
        linestyles='dashed'
    )

    cbar = plt.colorbar(PC)
    cbar.add_lines(contour)
    plt.xlabel('High dim distance >>')
    plt.ylabel('Low dim distance >>')
    plt.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        labeltop=False
    )
    plt.tick_params(
        axis='y',
        which='both',
        left=False,
        right=False,
        labelleft=False,
        labelright=False
    )
    plt.show()
    cbar.remove()



if __name__ == '__main__':
    # CHOOSE HOW TO GET LOW- AND HIGH-DIM DISTANCES #

    # Uncomment for exponentially growing distances
    low_dim_dists, high_dim_dists = get_exp_dists()

    # Uncomment for random multivariate-gaussian distances
    # low_dim_dists, high_dim_dists = get_gaussian_dists(
    #     high_dimensionality=50,
    #     low_dimensionality=2,
    #     num_points=20
    # )

    # Uncomment for MNIST distances
    # low_dim_dists, high_dim_dists = get_mnist_dists(num_points=20)

    # Uncomment this and below for linearly growing distances
    # low_dim_dists, high_dim_dists = get_simple_dists(upper_bound=10)

    # tSNE gradients plot
    gradient = tsne_grads(high_dim_dists, low_dim_dists)
    num_dists = len(low_dim_dists)
    make_plot(num_dists, gradient)

    # Uncomment this and above for linearly growing distances
    # low_dim_dists, high_dim_dists = get_simple_dists(upper_bound=1)

    # UMAP gradients plot
    a = 1
    b = 1
    k = 20
    gradient = umap_grads(high_dim_dists, low_dim_dists, a, b, k)
    num_dists = len(low_dim_dists)
    make_plot(num_dists, gradient)
