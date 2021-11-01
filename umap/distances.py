# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 3 clause
import numba
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def make_dist_plots(x_train, projection, y_train, alg_str):
    """
    Make histogram of distance relationships before and after projection
    """
    orig_dists = np.expand_dims(x_train, 0) - np.expand_dims(x_train, 1)
    orig_dists = np.sum(np.square(orig_dists), -1)
    orig_reshaped = np.reshape(orig_dists, -1)
    new_dists = np.expand_dims(projection, 0) - np.expand_dims(projection, 1)
    new_dists = np.sum(np.square(new_dists), -1)
    new_reshaped = np.reshape(new_dists, -1)

    orig_nn = np.argsort(orig_dists, axis=1)
    new_nn = np.argsort(new_dists, axis=1)

    indices = np.argsort(orig_reshaped)
    new_reshaped = new_reshaped[indices]
    N = int(new_reshaped.shape[0])
    num_points = int(orig_dists.shape[0])
    conv_size = 1000
    means = np.convolve(new_reshaped, np.ones(conv_size)/conv_size, mode='valid')

    plt.scatter(np.arange(N), new_reshaped, c='blue', s=0.01)
    plt.scatter(np.arange(int(means.shape[0])), means, c='red', s=0.1)
    plt.xlabel('Index after sorting by high-dim distance')
    plt.ylabel('Distance')
    redpatch = mpatches.Patch(color='red', label='Running average of size %d' % conv_size)
    bluepatch = mpatches.Patch(color='blue', label='Single points')
    plt.legend(handles=[redpatch, bluepatch])
    plt.title('Distances in %s projected space, sorted by high-dim distances' % alg_str)
    plt.savefig('images/sorted_distances_%s.png' % alg_str)
    
    plt.clf()
    plt.scatter(orig_reshaped[indices], new_reshaped, c='blue', s=0.01)
    plt.scatter(orig_reshaped[indices][:len(means)], means, c='red')
    plt.xlabel('High-dimensional distances')
    plt.ylabel('Low-dimensional distances')
    redpatch = mpatches.Patch(color='red', label='Running average of size %d' % conv_size)
    bluepatch = mpatches.Patch(color='blue', label='Single points')
    plt.legend(handles=[redpatch, bluepatch])
    plt.title('Low-dim distance vs. high-dim distance for %s' % alg_str)
    plt.savefig('images/distance_vs_distance_%s.png' % alg_str)

    # plt.clf()
    # overlap_ratio = []
    # for p in range(1, int(math.log2(num_points))):
    #     k = pow(2, p) - pow(2, p-1)
    #     k_orig_nn = orig_nn[:, int(pow(2, p-1)) : min(int(pow(2, p)), len(orig_nn[0]))]
    #     k_new_nn = new_nn[:, int(pow(2, p-1)) : min(int(pow(2, p)), len(new_nn[0]))]
    #     nn_overlap = np.zeros(k)
    #     for i in range(num_points):
    #         for j in range(k):
    #             if k_orig_nn[i, j] in k_new_nn[i]:
    #                 nn_overlap[j] += 1
    #     overlap = sum(nn_overlap) / (num_points * k)
    #     overlap_ratio.append(overlap)

    # plt.scatter(np.power(2, np.arange(len(overlap_ratio))), overlap_ratio, c='blue')
    # plt.title('Percent of exponential nearest neighbor overlap for %s' % alg_str)
    # plt.xlabel('Number of nearest neighbors being compared')
    # plt.ylabel('Percent overlap')
    # ax = plt.gca()
    # ax.set_ylim([0, 1])
    # plt.savefig('images/exponential_overlap_percent_%s.png' % alg_str)

    plt.clf()
    overlap_ratio = []
    k = 10
    for p in range(1, int(num_points/k)):
        k_orig_nn = orig_nn[:, int(k*(p-1)) : min(int(k*p), len(orig_nn[0]))]
        k_new_nn = new_nn[:, int(k*(p-1)) : min(int(k*p), len(new_nn[0]))]
        nn_overlap = np.zeros(k)
        for i in range(num_points):
            for j in range(k):
                if k_orig_nn[i, j] in k_new_nn[i]:
                    nn_overlap[j] += 1
        overlap = sum(nn_overlap) / (num_points * k)
        overlap_ratio.append(overlap)

    plt.scatter(np.arange(len(overlap_ratio)) * k, overlap_ratio, c='blue')
    plt.title('Percent of constant nearest neighbor overlap for %s' % alg_str)
    plt.xlabel('The [i, i + %d] nearest neighbors being compared' % k)
    plt.ylabel('Percent overlap on [i, i + %d] nearest neighbors' % k)
    ax = plt.gca()
    ax.set_ylim([0, 1])
    plt.savefig('images/constant_overlap_percent_%s.png' % alg_str)
    plt.clf()


@numba.njit(fastmath=True)
def euclidean(x, y):
    """Standard euclidean distance.

    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)


@numba.njit(fastmath=True)
def euclidean_grad(x, y):
    """Standard euclidean distance and its gradient.

    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
        \frac{dD(x, y)}{dx} = (x_i - y_i)/D(x,y)
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    d = np.sqrt(result)
    grad = (x - y) / (1e-6 + d)
    return d, grad

named_distances = {
    "euclidean": euclidean,
}

named_distances_with_gradients = {
    "euclidean": euclidean_grad,
}
