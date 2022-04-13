import os
import copy
import time
from tqdm import tqdm
import numpy as np
from experiment_utils.metrics import classifier_accuracy, cluster_quality, cluster_distances
from experiment_utils.general_utils import get_ab, make_plot
from experiment_utils.get_data import get_dataset
from experiment_utils.get_algorithm import get_algorithm

def cpu_analysis():
    datasets = [
        'mnist',
        'fashion_mnist',
        'cifar',
        'swiss_roll',
        'coil',
        'google_news',
    ]
    num_points_list = [
        60000,
        60000,
        50000,
        5000,
        7200,
        350000,
    ]

    modifiable_params = [
        'default',
        'neg_sample_rate',
        'random_init',
        'umap_metric',
        'tsne_symmetrization',
        'normalized',
        'sym_attraction',
        'frobenius',
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
            'normalized': True, # Also set amplify_grads to True
            'sym_attraction': False,
            'frobenius': False,
            'angular': False,
            'tsne_scalars': True,
            'gpu': False,
            'num_threads': -1,
            'numba': False
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
            'numba': False
        },
        'uniform_umap': {
            'optimize_method': 'uniform_umap',
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
            'numba': False
        },
        'uniform_umap_tsne': {
            'optimize_method': 'uniform_umap',
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
            'numba': False
        }
    }
    algorithms = list(default_params.keys())

    outputs_path = os.path.join('outputs', 'cpu')
    pbar = tqdm(enumerate(datasets), total=len(datasets))
    for data_i, dataset in pbar:
        try:
            points, labels = get_dataset(dataset, num_points_list[data_i])
        except Exception as E:
            print('Could not fine dataset %s' % dataset)
            print('Error raised was:', str(E))
            print('Continuing')
            print('.')
            print('.')
            print('.')
            print('\n')

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
                        # Only really applies for the original umap alg
                        if param == 'neg_sample_rate':
                            instance_params['neg_sample_rate'] = 1
                        else:
                            instance_params[param] = not instance_params[param]

                    # normalized and amplify_grads go together
                    instance_params['amplify_grads'] = instance_params['normalized']

                    # google-news dataset is cosine distance and too big for Lap. Eigenmaps
                    if dataset == 'google_news':
                        instance_params['random_init'] = True
                        instance_params['angular'] = True

                    instance_params['optimize_method'] = algorithm
                    a, b = get_ab(instance_params['tsne_scalars'])
                    instance_params['a'] = a
                    instance_params['b'] = b

                    dr = get_algorithm('uniform_umap', instance_params, verbose=False)

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
                    if num_points < 10000:
                        spectral_score = cluster_quality(embedding, labels, cluster_model='spectral')
                    else:
                        spectral_score = -1

                    metrics = {
                        # 'cluster_dists': cluster_distances(embedding, labels),
                        'knn_accuracy': classifier_accuracy(embedding, labels, 100),
                        'kmeans-v-score': kmeans_score,
                        'dbscan-v-score': dbscan_score,
                        'spectral-v-score': spectral_score
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


def gpu_analysis():
    datasets = [
        'mnist',
        'fashion_mnist',
        'swiss_roll',
        'coil',
        'google_news',
    ]
    num_points_list = [
        60000,
        60000,
        50000,
        5000,
        7200,
        350000,
    ]

    experiment_params = {
        'recreate_tsne_gpu': {
            'optimize_method': 'uniform_umap',
            'n_neighbors': 15,
            'random_init': False,
            'umap_metric': False,
            'tsne_symmetrization': False,
            'neg_sample_rate': 1,
            'n_epochs': 500,
            'normalized': True, # Also set amplify_grads to True
            'sym_attraction': False,
            'frobenius': False,
            'angular': False,
            'tsne_scalars': True,
            'gpu': True,
            'num_threads': -1,
            'numba': False
        },
        'recreate_umap_gpu': {
            'optimize_method': 'uniform_umap',
            'n_neighbors': 15,
            'random_init': False,
            'umap_metric': False,
            'tsne_symmetrization': False,
            'neg_sample_rate': 1,
            'n_epochs': 500,
            'normalized': False, # Also set amplify_grads to True
            'sym_attraction': False,
            'frobenius': False,
            'angular': False,
            'tsne_scalars': True,
            'gpu': True,
            'num_threads': -1,
            'numba': False
        },
        'recreate_tsne_gpu_frob': {
            'optimize_method': 'uniform_umap',
            'n_neighbors': 15,
            'random_init': False,
            'umap_metric': False,
            'tsne_symmetrization': False,
            'neg_sample_rate': 1,
            'n_epochs': 500,
            'normalized': True, # Also set amplify_grads to True
            'sym_attraction': False,
            'frobenius': True,
            'angular': False,
            'tsne_scalars': True,
            'gpu': True,
            'num_threads': -1,
            'numba': False
        },
        'recreate_umap_gpu_frob': {
            'optimize_method': 'uniform_umap',
            'n_neighbors': 15,
            'random_init': False,
            'umap_metric': False,
            'tsne_symmetrization': False,
            'neg_sample_rate': 1,
            'n_epochs': 500,
            'normalized': False, # Also set amplify_grads to True
            'sym_attraction': False,
            'frobenius': True,
            'angular': False,
            'tsne_scalars': True,
            'gpu': True,
            'num_threads': -1,
            'numba': False
        },

        ### FIXME FIXME FIXME -- RAPIDS UMAP

        ### FIXME FIXME FIXME -- RAPIDS TSNE
    }

    outputs_path = os.path.join('outputs', 'gpu')
    pbar = tqdm(enumerate(datasets), total=len(datasets))
    for data_i, dataset in pbar:
        try:
            points, labels = get_dataset(dataset, num_points)
        except Exception as E:
            print('Could not fine dataset %s' % dataset)
            print('Error raised was:', str(E))
            print('Continuing')
            print('.')
            print('.')
            print('.')
            print('\n')

        dataset_output_path = os.path.join(outputs_path, dataset)
        if not os.path.isdir(dataset_output_path):
            os.makedirs(dataset_output_path, exist_ok=True)

        for experiment in experiment_params:
            experiment_path = os.path.join(dataset_output_path, experiment)
            if not os.path.isdir(experiment_path):
                os.makedirs(experiment_path, exist_ok=True)
            try:
                instance_params = copy.copy(experiment_params[experiment])
                instance_params['amplify_grads'] = instance_params['normalized'] # normalized and amplify_grads go together

                # google-news dataset requires cosine distance and is too big for Lap. Eigenmaps initialization
                if dataset == 'google_news':
                    instance_params['random_init'] = True
                    instance_params['angular'] = True

                instance_params['a'] = 1
                instance_params['b'] = 1
                dr = get_algorithm('uniform_umap', instance_params, verbose=False)

                start = time.time()
                embedding = dr.fit_transform(points)
                end = time.time()
                total_time = end - start
                try:
                    opt_time = embedding.opt_time
                except AttributeError:
                    opt_time = -1

                times = {
                    'opt_time': opt_time,
                    'total_time': total_time
                }
                np.save(os.path.join(experiment_path, "times.npy"), times)
                np.save(os.path.join(experiment_path, "embedding.npy"), embedding)
                np.save(os.path.join(experiment_path, "labels.npy"), labels)
            except Exception as E:
                print('Could not run analysis for %s gpu experiment on %s dataset' % (experiment, dataset))
                print('The following exception was raised:')
                print(str(E))
                print('continuing')
                print('.')
                print('.')
                print('.')
                continue


def data_size_timings():
    datasets = [
        'mnist',
        'google_news',
    ]
    num_points_list = [
        1000,
        4000,
        16000,
        64000,
        256000,
        1024000,
    ]

    experiment_params = {
        'tsne': {
            'optimize_method': 'tsne',
            'n_neighbors': 90,
            'random_init': True,
            'umap_metric': False,
            'tsne_symmetrization': True,
            'neg_sample_rate': 1,
            'n_epochs': 500,
            'normalized': True, # Also set amplify_grads to True
            'sym_attraction': False,
            'frobenius': False,
            'angular': False,
            'tsne_scalars': True,
            'gpu': False,
            'num_threads': -1,
            'numba': False
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
            'numba': False
        },
        'uniform_umap': {
            'optimize_method': 'uniform_umap',
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
            'numba': False
        },
        'uniform_umap': {
            'optimize_method': 'uniform_umap',
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
            'gpu': True,
            'num_threads': -1,
            'numba': False
        }
    }

    outputs_path = os.path.join('outputs', 'dataset_size_timing')
    pbar = tqdm(enumerate(datasets), total=len(datasets))
    for data_i, dataset in pbar:
        dataset_output_path = os.path.join(outputs_path, dataset)
        if not os.path.isdir(dataset_output_path):
            os.makedirs(dataset_output_path, exist_ok=True)
        for num_points in num_points_list:
            try:
                points, labels = get_dataset(dataset, num_points)
            except Exception as E:
                print('Could not fine dataset %s' % dataset)
                print('Error raised was:', str(E))
                print('Continuing')
                print('.')
                print('.')
                print('.')
                print('\n')
            num_points_path = os.path.join(dataset_output_path, '%s_points' % str(num_points))
            if not os.path.isdir(num_points_path):
                os.makedirs(num_points_path, exist_ok=True)
            for experiment in experiment_params:
                experiment_path = os.path.join(num_points_path, experiment)
                if not os.path.isdir(experiment_path):
                    try:
                        os.makedirs(experiment_path, exist_ok=True)
                        instance_params = copy.copy(experiment_params[experiment])
                        instance_params['amplify_grads'] = instance_params['normalized'] # normalized and amplify_grads go together

                        # google-news dataset requires cosine distance and is too big for Lap. Eigenmaps initialization
                        if dataset == 'google_news':
                            instance_params['random_init'] = True
                            instance_params['angular'] = True
                            if experiment == 'tsne' and num_points > 250000:
                                raise ValueError("Too many points for Barnes-Hut TSNE")

                        instance_params['a'] = 1
                        instance_params['b'] = 1
                        dr = get_algorithm('uniform_umap', instance_params, verbose=False)

                        start = time.time()
                        embedding = dr.fit_transform(points)
                        end = time.time()
                        total_time = end - start
                        try:
                            opt_time = embedding.opt_time
                        except AttributeError:
                            opt_time = -1

                        times = {
                            'opt_time': opt_time,
                            'total_time': total_time
                        }
                        np.save(os.path.join(experiment_path, "times.npy"), times)
                    except Exception as E:
                        print('Could not run analysis for %s data_size experiment on %s dataset' % (experiment, dataset))
                        print('The following exception was raised:')
                        print(str(E))
                        print('continuing')
                        print('.')
                        print('.')
                        print('.')
                        continue


if __name__ == '__main__':
    # cpu_analysis()
    # gpu_analysis()
    data_size_timings()
