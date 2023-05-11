import os, csv, glob
import h5py
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.datasets import make_swiss_roll
from mnist import MNIST

def load_mnist():
    mnist_data_path = os.path.join('GDR', 'data', 'mnist')
    if not os.path.isdir(mnist_data_path):
        import subprocess
        subprocess.call(os.path.join('GDR', 'scripts', 'mnist_get_data.sh'))

    mndata = MNIST(mnist_data_path)
    points, labels = mndata.load_training()
    points = np.array(points)
    labels = np.array(labels)
    return points, labels

def load_google_news(num_points):
    # To run this dataset, download https://data.world/jaredfern/googlenews-reduced-200-d
    #   and place it into the directory 'data'
    with open(os.path.join('GDR', 'data', 'gnews_mod.csv'), 'r', encoding="utf-8") as f:
        reader = csv.reader(f)
        if num_points < 0:
            num_points = 350000
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
    return points, labels

# The single-cell data has many labels. Label 20 is the cell-cluster, which provides the cleanest
#   separation in the embedding
def load_single_cell_data(directory=None, class_dim=20):
    """
    This is using the single cell dataset available at https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE115746
    """
    # Single cell dataset is too large to pickle
    if directory is None:
        directory = os.path.join('GDR', 'data', 'single_cell')

    labels_file = os.path.join(directory, 'GSE115746_complete_metadata_28706-cells.csv')
    labels = []
    with open(labels_file, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            labels.append(line)
    labels = np.array(labels)
    label_dict = {}
    for label in labels:
        label_dict[label[0]] = label[class_dim]

    data_file = os.path.join(directory, 'GSE115746_cells_exon_counts.csv')
    if not os.path.exists(data_file):
        raise ValueError('Could not find single cell data!')

    data_size = 23178
    data = np.zeros([45769, data_size])
    class_labels = []
    with open(data_file, 'r') as f:
        reader = csv.reader(f)
        for i, line in tqdm(enumerate(reader), total=45769):
            if i == 0:
                for j, data_label in enumerate(line):
                    if j == data_size + 1:
                        break
                    if data_label in label_dict:
                        class_labels.append(label_dict[data_label])
                class_labels = np.array(class_labels)
            else:
                data[i] = np.array([int(l) for l in line[1:data_size+1]])
    points = np.transpose(data)
    unique_labels = np.unique(class_labels)
    labels = np.zeros(len(class_labels))
    for i in range(data_size):
        label_index = -1
        for j, unique_label in enumerate(unique_labels):
            if unique_label == class_labels[i]:
                label_index = j
                break
        assert label_index >= 0
        labels[i] = label_index

    return points, labels

def load_coil100_data(directory=None):
    """
    This is using the coil100 dataset available on Kaggle at https://www.kaggle.com/datasets/jessicali9530/coil100
    Using it requires manually unzipping it into a directory
    """
    if directory is None:
        directory = os.path.join('GDR', 'data', 'coil-100')
    pickled_path = os.path.join(directory, 'pickled_coil.npy')
    if os.path.exists(pickled_path):
        dataset = np.load(pickled_path, allow_pickle=True)[()]
        return dataset['points'], dataset['labels']

    print('Could not find pickled dataset at {}. Loading from png files and pickling...'.format(pickled_path))
    filelist = glob.glob(os.path.join(directory, '*.png'))
    if not filelist:
        raise ValueError('Coil 100 data directory {} is empty!'.format(directory))

    points = np.zeros([7200, 128, 128, 3])
    labels = np.zeros([7200])
    for i, fname in enumerate(filelist):
        image = np.array(Image.open(fname))
        points[i] = image

        image_name = os.path.split(fname)[-1]
        # This assumes that your images are named objXY__i.png
        #   where XY are the class label and i is the picture angle
        class_label = [int(c) for c in image_name[:6] if c.isdigit()]
        class_label = np.array(class_label[::-1])
        digit_powers = np.power(10, np.arange(len(class_label)))
        class_label = np.sum(class_label * digit_powers)
        labels[i] = class_label

    points = np.reshape(points, [7200, -1])
    np.save(pickled_path, {'points': points, 'labels': labels})
    return points, labels

def load_cifar_data(directory=None):
    """
    Using the cifar10 dataset found at https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    """
    def unpickle(data_path):
        with open(data_path, 'rb') as fo:
            d = pickle.load(fo, encoding='latin1')
        return d

    if directory is None:
        directory = os.path.join('GDR', 'data', 'cifar')

    if not os.path.isdir(directory):
        import subprocess
        subprocess.call(os.path.join('GDR', 'scripts', 'cifar_get_data.sh'))

    filelist = glob.glob(os.path.join(directory, 'data_batch_*'))
    if not filelist:
        raise ValueError('Cifar data directory {} is not in the expected format!'.format(directory))

    points = np.zeros([50000, 3072])
    labels = np.zeros([50000])
    for i in range(1, 6):
        data_batch_path = os.path.join(directory, 'data_batch_{}'.format(i))
        data_batch = unpickle(data_batch_path)
        points[10000*(i-1):10000*i] = data_batch['data']
        labels[10000*(i-1):10000*i] = data_batch['labels']

    return points, labels

def load_fashion_mnist_data(directory=None):
    """
    Using the Fashion MNIST dataset found at https://www.kaggle.com/datasets/zalando-research/fashionmnist
    """
    if directory is None:
        directory = os.path.join('GDR', 'data', 'fashion_mnist')

    images_file = os.path.join(directory, 'train-images-idx3-ubyte')
    with open(images_file, 'rb') as f:
        points = np.fromfile(f, dtype=np.uint8)
        points = points[16:].reshape(-1, 784).astype(np.float32)
    labels_file = os.path.join(directory, 'train-labels-idx1-ubyte')
    with open(labels_file, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        labels = labels[8:].astype(np.int32)

    return points, labels

def load_usps_data(directory=None):
    """ Found at https://www.kaggle.com/datasets/bistaumanga/usps-dataset?resource=download """
    if directory is None:
        directory = os.path.join('GDR', 'data', 'usps')
    path = os.path.join(directory, 'usps.h5')
    with h5py.File(path, 'r') as hf:
            train = hf.get('train')
            points = train.get('data')[:]
            labels = train.get('target')[:]
    return points, labels

def upsample_dataset(num_points, points, labels):
    # If the dataset doesn't have as many points as we want, make copies of the
    #   dataset until it does
    # Note -- this is only for timing purposes and resulting embeddings may be bogus
    assert int(points.shape[0]) == int(labels.shape[0])
    num_samples = int(points.shape[0])
    while num_points > num_samples:
        # Multiply by 2 on each dimension of the points when making copies of dataset
        #   - want to make sure that optimization doesn't get arbitrarily faster
        #     with identical copies of points
        points = np.concatenate([points, points*2], axis=0)[:num_points]
        labels = np.concatenate([labels, labels], axis=0)[:num_points]
        num_samples = int(points.shape[0])

    return points, labels

def resample_points(desired_size, points, labels):
    """
    If desired_size > len(points), upsample points until it is big enough
    Then sample down to get to the desired_size
    """
    points, labels = upsample_dataset(desired_size, points, labels)
    num_samples = int(points.shape[0])
    downsample_stride = int(float(num_samples) / desired_size)
    points, labels = points[::downsample_stride], labels[::downsample_stride]
    num_samples = int(points.shape[0])
    points = np.reshape(points, [num_samples, -1])

    return points, labels

def resample_dim(desired_dim, points):
    dim = int(points.shape[1])
    while dim < desired_dim:
        points = np.concatenate([points, points], axis=-1)
        dim = int(points.shape[1])
    random_perm = np.random.permutation(np.arange(dim))
    points = points[:, random_perm]
    points = points[:, :desired_dim]
    return points

def subsample_along_classes(points, labels, num_classes, points_per_class, class_list=[]):
    unique_classes = np.unique(labels)
    if num_classes > len(unique_classes):
        raise ValueError('Cannot subsample to {} classes when only have {} available'.format(num_classes, len(unique_classes)))
    if not class_list:
        all_classes = np.unique(labels)
        class_list = np.random.choice(all_classes, num_classes, replace=False)

    per_class_samples = [np.where(labels == sampled_class)[0] for sampled_class in class_list]
    min_per_class = min([len(s) for s in per_class_samples])
    per_class_samples = [s[:min_per_class] for s in per_class_samples]
    sample_indices = np.squeeze(np.stack([per_class_samples], axis=-1))
    total_points_per_class = int(sample_indices.shape[-1])
    if points_per_class < total_points_per_class:
        stride_rate = float(total_points_per_class) / points_per_class
        class_subsample_indices = np.arange(0, total_points_per_class, step=stride_rate).astype(np.int32)
        sample_indices = sample_indices[:, class_subsample_indices]

    sample_indices = np.reshape(sample_indices, -1)
    points = points[sample_indices]
    labels = labels[sample_indices]
    return points, labels


def get_dataset(data_name, num_points, normalize=True, desired_dim=-1):
    if data_name == 'mnist':
        points, labels = load_mnist()
    elif data_name == 'usps':
        points, labels = load_usps_data()
    elif data_name == 'fashion_mnist':
        points, labels = load_fashion_mnist_data()
    elif data_name == 'cifar':
        points, labels = load_cifar_data()
    elif data_name == 'swiss_roll':
        points, _ = make_swiss_roll(n_samples=num_points, noise=0.01)
        labels = np.ones(num_points)
    elif data_name == 'google_news':
        points, labels = load_google_news(num_points)
    elif data_name == 'coil':
        points, labels = load_coil100_data()
    elif data_name == 'single_cell':
        points, labels = load_single_cell_data()
    else:
        raise ValueError("Unsupported dataset")

    if desired_dim > 0:
        points = resample_dim(desired_dim, points)

    if num_points < 0:
        num_points = int(points.shape[0])
    points, labels = resample_points(num_points, points, labels)

    # FIXME - do we need this normalization?
    if normalize:
        points = points / np.max(points).astype(np.float32)

    return points, labels
