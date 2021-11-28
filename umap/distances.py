# Author: Leland McInnes <leland.mcinnes@gmail.com>
#
# License: BSD 3 clause
import warnings
import os
import numba
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def remove_diag(A):
    removed = A[~np.eye(A.shape[0],dtype=bool)].reshape(A.shape[0],int(A.shape[0])-1, -1)
    return np.squeeze(removed)

def make_dist_plots(x_train, projection, y_train, alg_str, dataset_str):
    """
    Make plots of distance relationships before and after projection
    """
    if len(x_train) > 1000:
        raise ValueError("Making plots requires too much memory. Increase the downsample-stride parameter")
    warnings.filterwarnings("ignore")

    print('Making plots of distance relationships...')
    print('\t- Calculating high and low dim pairwise distances...')
    image_dir = os.path.join('images', alg_str, dataset_str)
    os.makedirs(image_dir, exist_ok=True)
    orig_dists = remove_diag(np.expand_dims(x_train, 0) - np.expand_dims(x_train, 1))
    orig_dists = np.sum(np.square(orig_dists), -1)
    orig_reshaped = np.reshape(orig_dists, -1)
    new_dists = remove_diag(np.expand_dims(projection, 0) - np.expand_dims(projection, 1))
    new_dists = np.sum(np.square(new_dists), -1)
    new_reshaped = np.reshape(new_dists, -1)

    print('\t- Saving dim reduction output...')
    plt.scatter(projection[:, 0], projection[:, 1], c=y_train, s=1)
    img_path = os.path.join(image_dir, '%s_output.png' % alg_str)
    plt.savefig(img_path)

    plt.clf()
    print('\t- Saving distance vs distance plot...')
    indices = np.argsort(orig_reshaped)
    new_reshaped = np.abs(new_reshaped[indices])
    N = int(new_reshaped.shape[0])
    num_points = int(orig_dists.shape[0])
    conv_size = 1000
    means = np.convolve(new_reshaped, np.ones(conv_size)/conv_size, mode='valid')

    # High-dim distances on x axis (sorted by high-dim distances low -> high) 
    # Low dim distances on y axis
    plt.scatter(orig_reshaped[indices], new_reshaped, c='blue', s=0.01)
    plt.scatter(orig_reshaped[indices][:len(means)], means, c='red')
    plt.xlabel('High-dimensional distances')
    plt.ylabel('Low-dimensional distances')
    redpatch = mpatches.Patch(color='red', label='Running average of size %d' % conv_size)
    bluepatch = mpatches.Patch(color='blue', label='Single points')
    plt.legend(handles=[redpatch, bluepatch])
    plt.title('Low-dim distance vs. high-dim distance for %s' % alg_str)
    img_path = os.path.join(image_dir, 'distance_vs_distance_%s.png' % alg_str)
    plt.savefig(img_path)

    # Nat numbers on x axis
    # Low dim distances on y axis (sorted by high-dim distances low -> high)
    # This is just the y-values of the previous plot
    print('\t- Saving sorted distances plot...')
    plt.clf()
    plt.scatter(np.arange(N), new_reshaped, c='blue', s=0.01)
    plt.scatter(np.arange(int(means.shape[0])), means, c='red', s=0.1)
    plt.xlabel('Index after sorting by high-dim distance')
    plt.ylabel('Distance')
    redpatch = mpatches.Patch(color='red', label='Running average of size %d' % conv_size)
    bluepatch = mpatches.Patch(color='blue', label='Single points')
    plt.legend(handles=[redpatch, bluepatch])
    plt.title('Distances in %s projected space, sorted by high-dim distances' % alg_str)
    img_path = os.path.join(image_dir, 'sorted_distances_%s.png' % alg_str)
    plt.savefig(img_path)

    # Nat numbers on x axis
    # Relative error in distance on y axis
    #   - the y axis is (high_dim_distance - low_dim_distance) / high_dim_distance
    #   - both distance arrays are scaled to [0, 1] (since abs vals are relative)
    #   - avoid division by zero by substituting 1's in for the zeros
    print('\t- Saving relative error plot...')
    plt.clf()
    scaled_orig = orig_reshaped / np.max(orig_reshaped)
    scaled_new = new_reshaped / np.max(new_reshaped)

    # Avoid division by zero
    orig_no_zeros = np.copy(scaled_orig)
    orig_no_zeros[orig_no_zeros == 0] = 1

    relative_error = (scaled_orig - scaled_new) / orig_no_zeros
    rel_err_means = np.convolve(
        relative_error,
        np.ones(conv_size)/conv_size,
        mode='valid'
    )

    plt.scatter(np.arange(N), relative_error, c='blue', s=0.01)
    plt.scatter(np.arange(int(rel_err_means.shape[0])), rel_err_means, c='red', s=0.1)
    plt.xlabel('Index after sorting by high-dim distance')
    plt.ylabel('(D_h - D_l) / D_h')
    redpatch = mpatches.Patch(color='red', label='Running average of size %d' % conv_size)
    bluepatch = mpatches.Patch(color='blue', label='Single points')
    plt.legend(handles=[redpatch, bluepatch])

    # Set y axis limits to ignore outliers
    lower_thresh = np.sort(rel_err_means)[int(len(rel_err_means) * 0.01)]
    upper_thresh = np.sort(rel_err_means)[int(len(rel_err_means) * 0.99)]
    ax = plt.gca()
    ax.set_ylim([lower_thresh, upper_thresh])

    plt.title('Ratio or distances in high- and low-dim space for %s' % alg_str)
    img_path = os.path.join(image_dir, 'relative_error_%s.png' % alg_str)
    plt.savefig(img_path)

    # High-dim distance on x-axis
    # Absolute change in sort index on y-axis
    # We're basically looking at "Do relatively small distances stay relatively small?"
    print('\t- Saving change in sort index plot...')
    plt.clf()
    new_indices = np.argsort(new_reshaped)
    index_diff = np.abs(indices - new_indices)
    means = np.convolve(index_diff, np.ones(conv_size)/conv_size, mode='valid')
    plt.scatter(orig_reshaped[indices], index_diff, c='blue', s=0.01)
    plt.scatter(orig_reshaped[indices][:len(means)], means, c='red', s=0.1)
    plt.xlabel('Change in index as a function of high-dim distance')
    plt.ylabel('Absolute change in index')
    redpatch = mpatches.Patch(color='red', label='Running average of size %d' % conv_size)
    bluepatch = mpatches.Patch(color='blue', label='Single points')
    plt.legend(handles=[redpatch, bluepatch])
    plt.title('Effect of high-dim distance on low-dim distance sort position')
    img_path = os.path.join(image_dir, 'change_in_sort_index_%s.png' % alg_str)
    plt.savefig(img_path)

    # Center of nearest neighbor index limits
    # Percent of nearest neighbor overlap between high- and low-dimensions on y-axis
    # This can be interpreted as:
    #     For each point, the what percent of the [i - k/2, i + k/2]
    #     nearest neighbors in the high-dim space are also in the
    #     [i - k/2, i + k/2] nearest neighbors in the low-dim space
    print('\t- Saving nearest neighbor overlap plot...')
    plt.clf()
    orig_nn = np.argsort(orig_dists, axis=1)
    new_nn = np.argsort(new_dists, axis=1)
    overlap_ratio = []
    k = 10
    for p in range(int(k/2), int(num_points - k/2)):
        low = int(p - k/2)
        high = min(num_points, int(p + k/2))
        k_orig_nn = orig_nn[:, low:high]
        k_new_nn = new_nn[:, low:high]
        nn_overlap = np.zeros(k)
        for i in range(num_points):
            for j in range(k):
                if k_orig_nn[i, j] in k_new_nn[i]:
                    nn_overlap[j] += 1
        # maximum amount of overlap is num_points * k
        ratio = sum(nn_overlap) / (num_points * k)
        overlap_ratio.append(ratio)

    plt.scatter(np.arange(k/2, int(num_points - k/2)), overlap_ratio, c='blue')
    plt.title('Percent of nearest neighbor overlap for %s' % alg_str)
    plt.xlabel('The [i - %d, i + %d] nearest neighbors being compared' % (int(k/2), int(k/2)))
    plt.ylabel('Percent overlap on [i - %d, i + %d] nearest neighbors' % (int(k/2), int(k/2)))
    ax = plt.gca()
    ax.set_ylim([0, 1])
    img_path = os.path.join(image_dir, 'nn_overlap_%s.png' % alg_str)
    plt.savefig(img_path)
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
