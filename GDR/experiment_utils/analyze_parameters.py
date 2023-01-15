import os
import copy
import time
from tqdm import tqdm
import numpy as np
from GDR.experiment_utils.metrics import classifier_accuracy, cluster_quality, cluster_distances
from GDR.experiment_utils.general_utils import get_ab, make_plot
from GDR.experiment_utils.get_data import get_dataset
from GDR.experiment_utils.get_algorithm import get_algorithm

def analyze_params():
    datasets = [
        # 'mnist',
        # 'fashion_mnist',
        # 'cifar',
        # 'swiss_roll',
        # 'coil',
        'single_cell',
        # 'google_news',
    ]
    num_points_list = [
        # 60000,
        # 60000,
        # 50000,
        # 5000
        # 7200,
        2300,
        # 350000,
    ]

    modifiable_params = [
        'default',
        'random_init',
        'umap_metric',
        'tsne_symmetrization',
        'normalized',
        'sym_attraction',
        'tsne_scalars'
    ]

    default_params = {
        'tsne': {
            'optimize_method': 'tsne',
            'n_neighbors': 90,
            'random_init': True,
            'umap_metric': False,
            'tsne_symmetrization': True,
            'neg_sample_rate': 1,
            'n_epochs': 500,
            'normalized': True,
            'sym_attraction': False,
            'frobenius': False,
            'angular': False,
            'tsne_scalars': True,
            'gpu': False,
            'num_threads': -1,
            'amplify_grads': True,
            'accelerated': False,
            'torch': False,
            'cython': True
        },
        'umap': {
            'optimize_method': 'umap',
            'n_neighbors': 15,
            'random_init': False,
            'umap_metric': True,
            'tsne_symmetrization': False,
            'n_epochs': 500,
            'neg_sample_rate': 5,
            'normalized': False,
            'sym_attraction': True,
            'frobenius': False,
            'angular': False,
            'tsne_scalars': False,
            'gpu': False,
            'num_threads': -1,
            'amplify_grads': False,
            'accelerated': False,
            'torch': False,
            'cython': True
        },
        'gdr': {
            'optimize_method': 'gdr',
            'n_neighbors': 15,
            'random_init': False,
            'umap_metric': False,
            'tsne_symmetrization': False,
            'n_epochs': 500,
            'neg_sample_rate': 1,
            'normalized': False,
            'sym_attraction': False,
            'frobenius': False,
            'angular': False,
            'tsne_scalars': True,
            'gpu': False,
            'num_threads': -1,
            'amplify_grads': False,
            'accelerated': False,
            'torch': False,
            'cython': True
        },
        'gdr_normed': {
            'optimize_method': 'gdr',
            'n_neighbors': 15,
            'random_init': False,
            'umap_metric': False,
            'tsne_symmetrization': False,
            'n_epochs': 500,
            'neg_sample_rate': 1,
            'normalized': True,
            'sym_attraction': False,
            'frobenius': False,
            'angular': False,
            'tsne_scalars': True,
            'gpu': False,
            'num_threads': -1,
            'amplify_grads': True,
            'accelerated': False,
            'torch': False,
            'cython': True
        }
    }
    algorithms = list(default_params.keys())

    outputs_path = os.path.join('outputs', 'cpu')
    pbar = tqdm(enumerate(datasets), total=len(datasets))
    for data_i, dataset in pbar:
        try:
            points, labels = get_dataset(dataset, num_points_list[data_i])
        except Exception as E:
            print('Could not find dataset %s' % dataset)
            print('Error raised was:', str(E))
            print('Continuing')
            print('.')
            print('.')
            print('.')
            print('\n')
            continue

        num_points = int(points.shape[0])
        dataset_output_path = os.path.join(outputs_path, dataset)
        if not os.path.isdir(dataset_output_path):
            os.makedirs(dataset_output_path, exist_ok=True)

        for algorithm in algorithms:
            alg_dataset_path = os.path.join(dataset_output_path, algorithm)
            if not os.path.isdir(alg_dataset_path):
                os.makedirs(alg_dataset_path, exist_ok=True)

            for param in modifiable_params:
                try:
                    param_path = os.path.join(alg_dataset_path, param)
                    if not os.path.isdir(param_path):
                        os.makedirs(param_path, exist_ok=True)

                    pbar.set_description('{} param for {} algorithm on {} dataset'.format(param, algorithm, dataset))

                    instance_params = copy.copy(default_params[algorithm])
                    if param != 'default':
                        # Neg sample rate isn't a bool, so treat it differently
                        # Only applies for the original umap alg
                        if param == 'neg_sample_rate':
                            instance_params['neg_sample_rate'] = 1
                        else:
                            instance_params[param] = not instance_params[param]

                    # google-news dataset is cosine distance and too big for Lap. Eigenmaps
                    if dataset == 'google_news':
                        instance_params['random_init'] = True
                        instance_params['angular'] = True
                        if algorithm == 'tsne':
                            print('Not running TSNE on Google News!')
                            continue

                    # instance_params['optimize_method'] = algorithm
                    a, b = get_ab(instance_params['tsne_scalars'])
                    instance_params['a'] = a
                    instance_params['b'] = b

                    dr = get_algorithm('gdr', instance_params, verbose=False)

                    start = time.time()
                    embedding = dr.fit_transform(points)
                    end = time.time()
                    total_time = end - start
                    try:
                        opt_time = embedding.opt_time
                    except AttributeError:
                        opt_time = -1

                    make_plot(
                        embedding,
                        labels,
                        save_path=os.path.join(param_path, 'embedding.png')
                    )
                    kmeans_score = cluster_quality(embedding, labels, cluster_model='kmeans')
                    dbscan_score = cluster_quality(embedding, labels, cluster_model='dbscan')

                    metrics = {
                        # 'cluster_dists': cluster_distances(embedding, labels),
                        'knn_accuracy': classifier_accuracy(embedding, labels, 100),
                        'kmeans-v-score': kmeans_score,
                        'dbscan-v-score': dbscan_score,
                    }
                    times = {
                        'opt_time': opt_time,
                        'total_time': total_time
                    }
                    np.save(os.path.join(param_path, "metrics.npy"), metrics)
                    np.save(os.path.join(param_path, "times.npy"), times)
                    np.save(os.path.join(param_path, "embedding.npy"), embedding)
                    np.save(os.path.join(param_path, "labels.npy"), labels)
                except Exception as E:
                    print('Could not run analysis for %s cpu algorithm on %s dataset' % (algorithm, dataset))
                    print('The following exception was raised:')
                    print(str(E))
                    raise(E)
                    print('continuing')
                    print('.')
                    print('.')
                    print('.')
                    continue


if __name__ == '__main__':
    analyze_params()
