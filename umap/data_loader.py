import numpy as np
from tensorflow import keras as tfk
import tensorflow_datasets as tfds

# FIXME -- other datasets should include
#   - SVHN
#   - TIMIT
#   - NORB
#   - CIFAR embeddings
def get_dataset(dataset_name, num_points):
    if dataset_name == 'mnist':
        train, _ = tfk.datasets.mnist.load_data(path='mnist.npz')
        points, labels = train
        points = np.array(points) / 255.0
    elif dataset_name == 'cifar':
        train, _ = tfk.datasets.cifar10.load_data()
        points, labels = train
    elif dataset_name == 'swiss_roll':
        num_samples = 50000
        points, _ = make_swiss_roll(n_samples=num_samples, noise=0.01)
        labels = np.arange(num_samples)
    elif dataset_name == 'google_news':
        import os, csv
        # To run this dataset, download https://data.world/jaredfern/googlenews-reduced-200-d
        #   and place it into the directory 'datasets'
        file = open(os.path.join('datasets', 'gnews_mod.csv'), 'r')
        reader = csv.reader(file)
        num_points = min(num_points, 350000)
        points = np.zeros([num_points, 200])
        for i, line in tqdm(enumerate(reader), total=num_points):
            # First line is column descriptions
            if i == 0:
                continue
            if i > num_points:
                break
            for j, element in enumerate(line[1:]): # First column is string text
                points[i-1, j] = float(element)
        labels = np.ones([num_points])
    elif dataset_name == 'coil':
        num_samples = 7200
        dataset, _ = tfds.load(
            'coil100',
            split=['train'],
            with_info=True
        )
        dataset = dataset[0].batch(1)
        points = np.zeros([num_samples, 128, 128, 3])
        labels = np.zeros([num_samples])
        for i, element in enumerate(dataset):
            points[i] = element['image']
            labels[i] = element['angle_label']
    else:
        raise ValueError("Unsupported dataset")

    num_samples = int(points.shape[0])
    if num_points > num_samples:
        print('WARNING: requested %d samples but dataset only has %d elements' % \
              (num_points, num_samples))
    downsample_stride = int(float(num_samples) / num_points)
    points, labels = points[::downsample_stride], labels[::downsample_stride]
    num_samples = int(points.shape[0])
    points = np.reshape(points, [num_samples, -1])

    # FIXME - CHECK NORMALIZATIONS??

    return points, labels
