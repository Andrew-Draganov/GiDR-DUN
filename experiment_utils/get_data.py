import os, csv
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.datasets import make_swiss_roll
from mnist import MNIST
#from tensorflow import keras as tfk
#import tensorflow_datasets as tfds

def load_mnist():
    mnist_data_path = os.path.join('data', 'mnist')
    if not os.path.isdir(mnist_data_path):
        import subprocess
        subprocess.call(os.path.join('utils', 'mnist_get_data.sh'))

    mndata = MNIST(mnist_data_path)
    points, labels = mndata.load_training()
    points = np.array(points)
    labels = np.array(labels)
    return points, labels

def upsample_dataset(num_points, points, labels):
    # If the dataset doesn't have as many points as we want, make copies of the
    #   dataset until it does
    # Note -- this is only for timing purposes and resulting embeddings may be bogus
    assert int(points.shape[0]) == int(labels.shape[0])
    num_samples = int(points.shape[0])
    while num_points > num_samples:
        # add 1 to each dimension of the points when making copies of dataset
        #   - want to make sure that optimization doesn't get arbitrarily faster
        #     with identical copies of points
        points = np.concatenate([points, points+1], axis=0)
        labels = np.concatenate([labels, labels], axis=0)
        num_samples = int(points.shape[0])
        
    return points, labels

def get_dataset(data_name, num_points, normalize=True):
    if data_name == 'mnist':
        points, labels = load_mnist()
    elif data_name == 'fashion_mnist':
        train, _ = tfk.datasets.fashion_mnist.load_data()
        points, labels = train
    elif data_name == 'cifar':
        train, _ = tfk.datasets.cifar10.load_data()
        points, labels = train
    elif data_name == 'swiss_roll':
        num_samples = 50000
        points, _ = make_swiss_roll(n_samples=num_samples, noise=0.01)
        labels = np.arange(num_samples)
    elif data_name == 'google_news':
        # To run this dataset, download the dataset from https://data.world/jaredfern/googlenews-reduced-200-d/
        #   and place it into the directory 'data'
        file = open(os.path.join('data', 'gnews_mod.csv'), 'r')
        reader = csv.reader(file)
        num_points = min(num_points, 350000)
        if num_points < 0:
            num_points = 350000
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
    elif data_name == 'coil':
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
            labels[i] = element['object_id']
    else:
        raise ValueError("Unsupported dataset")

    if num_points < 0:
        num_points = int(points.shape[0])
    points, labels = upsample_dataset(num_points, points, labels)
    num_samples = int(points.shape[0])
    downsample_stride = int(float(num_samples) / num_points)
    points, labels = points[::downsample_stride], labels[::downsample_stride]
    num_samples = int(points.shape[0])
    points = np.reshape(points, [num_samples, -1])

    # FIXME - do we need this normalization?
    if normalize:
        points = np.array(points) / np.max(points).astype(np.float32)

    return points, labels
