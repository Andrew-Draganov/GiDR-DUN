import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

# Low- and High-Dim distance getters
def get_matrix_pca_dists():
    low_dim_points = np.arange(0.01, 0.5, 0.01)
    low_dim_points = np.expand_dims(low_dim_points, 1)
    low_dim_dists = get_pairwise_dist_matrix(low_dim_points)
    low_dim_dists = np.reshape(low_dim_dists, -1)

    high_dim_points = np.arange(0.01, 0.5, 0.01)
    high_dim_points = np.expand_dims(high_dim_points, 1)
    high_dim_dists = get_pairwise_dist_matrix(high_dim_points)
    high_dim_dists = np.reshape(high_dim_dists, -1)

    return low_dim_dists, high_dim_dists, len(low_dim_points)

def get_pairwise_dist_matrix(points):
    # FIXME -- just don't sqrt here and remove them from all grads calcs
    num_points = int(points.shape[0])
    dists = np.zeros([num_points, num_points])
    for i, x_i in enumerate(points):
        for j, x_j in enumerate(points):
            if i != j:
                dists[i, j] = np.sqrt(np.sum(np.square(x_i - x_j)))
    return dists

def get_linear_progression(upper_bound):
    low_dim_dists = np.arange(0.01, upper_bound, upper_bound/100)
    high_dim_dists = np.arange(0.01, upper_bound, upper_bound/100)
    return low_dim_dists, high_dim_dists

def get_exp_progression():
    low_dim_dists = [0.005 * 1.12 ** i for i in range(100)]
    high_dim_dists = [0.005 * 1.12 ** i for i in range(100)]
    return low_dim_dists, high_dim_dists

def get_pairwise_dist_array(points):
    dists = []
    for i, x_i in enumerate(points):
        for x_j in points[i:]:
            if not np.all(x_i == x_j):
                dists.append(np.square(np.sum(np.square(x_i - x_j))))
    return np.sort(dists)

def get_random_low_dim_dists(low_dimensionality, num_points):
    low_dim_points = np.random.multivariate_normal(
        np.zeros(low_dimensionality),
        np.eye(low_dimensionality),
        size=num_points
    )
    return get_pairwise_dist_array(low_dim_points)

def get_gaussian_dists(high_dimensionality, low_dimensionality, num_points):
    high_dim_points = np.random.multivariate_normal(
        np.zeros(high_dimensionality),
        np.eye(high_dimensionality),
        size=num_points
    )
    high_dim_dists = get_pairwise_dist_array(high_dim_points)
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
    high_dim_dists = get_pairwise_dist_array(points)
    low_dim_dists = get_random_low_dim_dists(2, num_points)

    return low_dim_dists, high_dim_dists


# GRADIENT DERIVATIONS
def pca_kernel(Dx, Dy, num_points):
    Kern_x = np.reshape(Dx, [num_points, num_points])
    Kern_x = np.square(Kern_x)
    # Zero row/col mean for PCA grads
    Kern_x -= np.mean(Kern_x, axis=0, keepdims=True)
    Kern_x -= np.mean(Kern_x, axis=1, keepdims=True)

    # Zero row/col mean for PCA grads
    Kern_y = np.reshape(Dy, [num_points, num_points])
    Kern_y = np.square(Kern_y)
    Kern_y -= np.mean(Kern_y, axis=0, keepdims=True)
    Kern_y -= np.mean(Kern_y, axis=1, keepdims=True)

    return Kern_x, Kern_y

def centered_pca_kernels(Dx, Dy, num_points):
    scalar = -4
    x_sort_indices = np.argsort(np.reshape(Dx, -1))
    y_sort_indices = np.argsort(np.reshape(Dy, -1))

    P, Q = pca_kernel(Dx, Dy, num_points)
    P = np.reshape(P, -1)[x_sort_indices]
    Q = np.reshape(Q, -1)[y_sort_indices]
    P, Q = np.meshgrid(P, Q)

    return P, Q

def pca_grads(Dx, Dy, num_points=-1, centered=False):
    scalar = -4
    if centered:
        assert num_points > 0
        P, Q = centered_pca_kernels(Dx, Dy, num_points)
    else:
        P, Q = np.meshgrid(np.square(Dx), np.square(Dy))
    Dy_stack = [Dy for _ in Dy]
    gradient = scalar * (Q - P) * np.stack(Dy_stack, -1)
    gradient = np.flip(gradient, 0)
    return gradient

def tsne_grads(Dx, Dy):
    P = np.exp(-(np.square(Dx) / 2))
    Q = 1 / (1 + np.square(Dy))
    P, Q = np.meshgrid(P, Q)
    Z_stack = [1 / (1 + np.square(Dy)) for _ in Dy]
    # FIXME - Is it okay to use the same distance values for both the kernels and the (y_i - y_j) scaling?
    Dy_stack = [Dy for _ in Dy]
    gradient = 4 * (P - Q) * np.stack(Z_stack, -1) * np.stack(Dy_stack, -1)
    gradient = np.flip(gradient, 0)
    return gradient

def frob_tsne_grads(Dx, Dy):
    P = np.exp(-(np.square(Dx) / 2))
    Q = 1 / (1 + np.square(Dy))
    P, Q = np.meshgrid(P, Q)
    Z_stack = [1 / (1 + np.square(Dy)) for _ in Dy]
    # FIXME - Is it okay to use the same distance values for both the kernels and the (y_i - y_j) scaling?
    Dy_stack = [Dy for _ in Dy]
    gradient = 4 * (P - Q) * (Q**2 + 2*Q**3) * np.stack(Z_stack, -1) * np.stack(Dy_stack, -1)
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

def frob_umap_grads(Dx, Dy, a, b, k):
    sigma = find_sigma(Dx, np.log2(k))
    P_vec = np.exp(-1 * (np.square(Dx) - np.min(np.square(Dx))) / sigma)
    Q_vec = 1 / (1 + a * np.square(Dy) ** b)
    attr_vec = P_vec * Q_vec * Q_vec
    rep_vec = Q_vec * Q_vec * Q_vec

    rep, attr = np.meshgrid(attr_vec, rep_vec)
    Dy_stack = [Dy for _ in Dy]
    gradient = (rep - attr) * np.stack(Dy_stack, -1)
    gradient = np.flip(gradient, 0)
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


def make_plot(
    low_dim_dists,
    high_dim_dists,
    gradient,
    upper_bound,
    contour=True
):
    num_dists = len(low_dim_dists)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    X, Y = np.meshgrid(np.arange(num_dists), np.arange(num_dists))
    ax1.matshow(gradient, cmap=plt.get_cmap('viridis'))
    PC = ax1.pcolor(X, Y, gradient)
    cbar = plt.colorbar(PC)

    if contour:
        contour = ax1.contour(
            X,
            Y,
            gradient,
            levels=[0, 10],
            colors='black',
            linestyles='dashed'
        )
        cbar.add_lines(contour)
    plt.xlabel('High dim distance >>')
    plt.ylabel('Low dim distance >>')

    x_tick_indices = np.arange(
        float(num_dists)/5,
        num_dists,
        float(num_dists)/5
    )

    y_tick_indices = np.arange(
        float(num_dists)/5,
        num_dists,
        float(num_dists)/5
    )

    plt.xticks(
        x_tick_indices,
        np.arange(
            float(upper_bound)/5,
            upper_bound,
            float(upper_bound)/5
        ).astype(np.int32)
    )
    plt.tick_params(
        axis='x',
        bottom=True,
        labelbottom=True,
        top=False,
        labeltop=False,
        which='both',
    )
    plt.yticks(
        y_tick_indices,
        np.arange(
            float(upper_bound)/5,
            upper_bound,
            float(upper_bound)/5
        )[::-1].astype(np.int32)
    )
    plt.tick_params(
        axis='y',
        left=True,
        labelleft=True,
        which='both',
    )
    plt.show()
    cbar.remove()


if __name__ == '__main__':
    upper_bound = 5
    # Basic PCA gradients plot
    # low_dim_dists, high_dim_dists = get_linear_progression(upper_bound=0.5)
    # gradient = pca_grads(high_dim_dists, low_dim_dists, centered=False)
    # make_plot(low_dim_dists, high_dim_dists, gradient, contour=True)

    # Centered PCA gradients plot
    # low_dim_dists, high_dim_dists, num_points = get_matrix_pca_dists()
    # gradient = pca_grads(high_dim_dists, low_dim_dists, num_points, centered=True)
    # make_plot(low_dim_dists, high_dim_dists, gradient, contour=False)

    # CHOOSE HOW TO GET LOW- AND HIGH-DIM DISTANCES #

    # Uncomment for random multivariate-gaussian distances
    # low_dim_dists, high_dim_dists = get_gaussian_dists(
    #     high_dimensionality=50,
    #     low_dimensionality=2,
    #     num_points=20
    # )

    # Uncomment for MNIST distances
    # from tensorflow import keras as tfk
    # low_dim_dists, high_dim_dists = get_mnist_dists(num_points=20)

    # Uncomment this and below for linearly growing distances
    # low_dim_dists, high_dim_dists = get_linear_progression(upper_bound=10)

    # tSNE gradients plot
    # gradient = tsne_grads(high_dim_dists, low_dim_dists)
    # make_plot(low_dim_dists, high_dim_dists, gradient)

    # Uncomment this and above for linearly growing distances
    low_dim_dists, high_dim_dists = get_linear_progression(upper_bound=upper_bound)

    # Uncomment for exponentially growing distances
    # low_dim_dists, high_dim_dists = get_exp_progression()

    # UMAP gradients plot
    a = 1
    b = 1
    k = 20
    gradient = tsne_grads(high_dim_dists, low_dim_dists)
    # gradient = umap_grads(high_dim_dists, low_dim_dists, a, b, k)
    make_plot(low_dim_dists, high_dim_dists, gradient, upper_bound)

    # Frobenius UMAP gradients plot
    a = 1
    b = 1
    k = 20
    gradient = frob_tsne_grads(high_dim_dists, low_dim_dists)
    # gradient = frob_umap_grads(high_dim_dists, low_dim_dists, a, b, k)
    make_plot(low_dim_dists, high_dim_dists, gradient, upper_bound)
