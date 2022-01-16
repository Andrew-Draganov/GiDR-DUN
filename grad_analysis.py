import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
# from tensorflow import keras as tfk

# Low- and High-Dim distance getters
def get_simple_dists():
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




# Kernels and likelihood calculations
def get_scalar(dists, low_dim=True):
    if low_dim:
        kernel = low_dim_kernel
    else:
        kernel = high_dim_kernel
    kernel_vals = kernel(dists)
    return np.sum(kernel_vals)

def umap_attraction_grad(low_dim_dists, high_dim_dists, a, b):
    grad_scalar = -2.0 * a * b * np.power(low_dim_dists, b - 1.0)
    grad_scalar /= a * np.power(low_dim_dists, b) + 1.0
    return grad_scalar * high_dim_kernel(high_dim_dists, remove_min=True)

def umap_repulsion_grad(low_dim_dists, high_dim_dists, a, b):
    phi_ijZ = 2.0 * b
    phi_ijZ /= (0.001 + low_dim_dists) * (a * np.power(low_dim_dists, b) + 1)
    return phi_ijZ * (1 - high_dim_kernel(high_dim_dists, remove_min=True))

def umap_grad(low_dim_dists, high_dim_dists, a, b):
    attr = umap_attraction_grad(low_dim_dists, high_dim_dists, a, b)
    rep = umap_repulsion_grad(low_dim_dists, high_dim_dists, a, b)
    return attr + rep

def low_dim_kernel(dists, a=1, b=1):
    return 1 / (1 + a * np.power(dists, b))

def high_dim_kernel(dists, remove_min=False):
    # FIXME - what would removing the min do?
    # FIXME - do sigma calculation binary search correctly
    psi = 0
    if remove_min:
        psi = np.min(dists)
    return np.exp(-(dists - psi)/ (2 * np.sqrt(np.var(dists))))

def tsne_grad(low_dim_dists, high_dim_dists, P, Z):
    q_kern = low_dim_kernel(low_dim_dists, 1, 1)
    p_kern = high_dim_kernel(high_dim_dists)
    return (p_kern / P - q_kern / Z) * q_kern





if __name__ == '__main__':
    # CHOOSE HOW TO GET LOW- AND HIGH-DIM DISTANCES #
    # low_dim_dists, high_dim_dists = get_simple_dists()
    # low_dim_dists, high_dim_dists = get_gaussian_dists(
    #     high_dimensionality=50,
    #     low_dimensionality=2,
    #     num_points=50
    # )
    # low_dim_dists, high_dim_dists = get_mnist_dists(num_points=100)

    # low_dim_dists = np.array(low_dim_dists)
    # high_dim_dists = np.array(high_dim_dists)
    # a = 1
    # b = 1
    # low_dim_grid, high_dim_grid = np.meshgrid(low_dim_dists, high_dim_dists)

    # Z = get_scalar(high_dim_dists, low_dim=True)
    # P = get_scalar(high_dim_dists, low_dim=False)
    # tsne_grads = tsne_grad(low_dim_grid, high_dim_grid, P, Z)
    # plt.matshow(tsne_grads, cmap=plt.get_cmap('viridis'))

    # tSNE original plot gradients
    D = np.arange(0.01, 10, 0.01)
    P = np.exp(-(np.square(D) / 2))
    Q = 1 / (1 + np.square(D))
    P, Q = np.meshgrid(P, Q)
    Z_stack = [1 / (1 + np.square(D)) for _ in D]
    D_stack = [D for _ in D]
    gradient = 4 * (P - Q) * np.stack(Z_stack, -1) * np.stack(D_stack, -1)
    gradient = np.flip(gradient, 0)
    plt.matshow(gradient, cmap=plt.get_cmap('viridis'))

    plt.colorbar()
    plt.xlabel('High dim distance >>')
    plt.ylabel('Low dim distance >>')
    # plt.tick_params(
    #     axis='x',
    #     which='both',
    #     bottom=False,
    #     top=False,
    #     labelbottom=False,
    #     labeltop=False
    # )
    # plt.tick_params(
    #     axis='y',
    #     which='both',
    #     left=False,
    #     right=False,
    #     labelleft=False,
    #     labelright=False
    # )
    plt.show()

    # UMAP
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


    a = 1
    b = 1
    k = 20
    D = np.arange(0.01, 1, 0.01)
    sigma = find_sigma(D, np.log2(k))
    P_vec = np.exp(-1 * (np.square(D) - np.min(np.square(D))) / sigma)
    attr_vec = -2 * a * b * np.power(np.square(D), b - 1) / (1 + np.square(D))
    rep_vec = 2 * b / ((0.001 + np.square(D)) * (1 + a * np.power(np.square(D), b)))

    P, attr = np.meshgrid(P_vec, attr_vec)
    _, rep = np.meshgrid(P_vec, rep_vec)
    print(np.max(rep))
    print(np.max(attr))
    D_stack = [D for _ in D]
    grad = (P * attr + (1 - P) * rep) * np.stack(D_stack, -1)
    grad = -1 * np.flip(grad, 0)
    plt.matshow(grad, cmap=plt.get_cmap('viridis'))


    # umap_grads = umap_grad(low_dim_grid, high_dim_grid, a, b)
    # plt.matshow(umap_grads, cmap=plt.get_cmap('viridis'), norm=matplotlib.colors.LogNorm())

    plt.colorbar()
    plt.xlabel('High dim distance >>')
    plt.ylabel('Low dim distance >>')
    # plt.tick_params(
    #     axis='x',
    #     which='both',
    #     bottom=False,
    #     top=False,
    #     labelbottom=False,
    #     labeltop=False
    # )
    # plt.tick_params(
    #     axis='y',
    #     which='both',
    #     left=False,
    #     right=False,
    #     labelleft=False,
    #     labelright=False
    # )
    plt.show()
