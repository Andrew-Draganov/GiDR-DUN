import argparse
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
            'normalized': True,
            'sym_attraction': False,
            'frobenius': False,
            'angular': False,
            'tsne_scalars': True,
            'gpu': False,
            'torch': False,
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
            'torch': False,
            'num_threads': -1,
            'numba': True
        },
        'gidr_dun': {
            'optimize_method': 'gidr_dun',
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
            'torch': False,
            'num_threads': -1,
            'numba': True
        },
        'gidr_dun_tsne': {
            'optimize_method': 'gidr_dun',
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
            'torch': False,
            'num_threads': -1,
            'numba': True
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
                        if algorithm == 'tsne':
                            print('Not running TSNE on Google News!')
                            continue

                    a, b = get_ab(instance_params['tsne_scalars'])
                    instance_params['a'] = a
                    instance_params['b'] = b
                    if algorithm == 'tsne':
                        method = 'original_tsne'
                    else:
                        method = 'gidr_dun'

                    dr = get_algorithm(method, instance_params, verbose=False)

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



def dim_timings():
    datasets = [
        'mnist',
    ]
    num_points_list = [
        60000,
    ]
    dims_list = [
        100,
        400,
        1600,
        3200,
        12800,
        51200
    ]

    experiment_params = {
        'gidr_dun': {
            'optimize_method': 'gidr_dun',
            'n_neighbors': 15,
            'random_init': False,
            'umap_metric': False,
            'tsne_symmetrization': False,
            'neg_sample_rate': 1,
            'n_epochs': 500,
            'normalized': False,
            'sym_attraction': False,
            'frobenius': False,
            'angular': False,
            'tsne_scalars': True,
            'gpu': False,
            'torch': False,
            'num_threads': -1,
            'numba': True
        },
        'gidr_dun_tsne': {
            'optimize_method': 'gidr_dun',
            'n_neighbors': 15,
            'random_init': False,
            'umap_metric': False,
            'tsne_symmetrization': False,
            'neg_sample_rate': 1,
            'n_epochs': 500,
            'normalized': True,
            'sym_attraction': False,
            'frobenius': False,
            'angular': False,
            'tsne_scalars': True,
            'gpu': False,
            'torch': False,
            'num_threads': -1,
            'numba': True
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
            'torch': False,
            'num_threads': -1,
            'numba': True
        },
    }

    outputs_path = os.path.join('outputs', 'dim_timing')
    pbar = tqdm(enumerate(datasets), total=len(datasets))
    for data_i, dataset in pbar:
        dataset_output_path = os.path.join(outputs_path, dataset)
        if not os.path.isdir(dataset_output_path):
            os.makedirs(dataset_output_path, exist_ok=True)
        for dim in dims_list:
            try:
                num_points = num_points_list[data_i]
                points, labels = get_dataset(dataset, num_points, desired_dim=dim)
            except Exception as E:
                print('Could not find dataset %s' % dataset)
                print('Error raised was:', str(E))
                print('Continuing')
                print('.')
                print('.')
                print('.')
                print('\n')
            dim_path = os.path.join(dataset_output_path, '%s_dim' % str(dim))
            if not os.path.isdir(dim_path):
                os.makedirs(dim_path, exist_ok=True)
            for experiment in experiment_params:
                experiment_path = os.path.join(dim_path, experiment)
                if not os.path.isdir(experiment_path):
                    try:
                        os.makedirs(experiment_path, exist_ok=True)
                        print(experiment_path)

                        instance_params = copy.copy(experiment_params[experiment])
                        instance_params['amplify_grads'] = instance_params['normalized'] # normalized and amplify_grads go together
                        if dim > 1000:
                            instance_params['random_init'] = True
                        a, b = get_ab(instance_params['tsne_scalars'])
                        instance_params['a'] = a
                        instance_params['b'] = b
                        dr = get_algorithm('gidr_dun', instance_params, verbose=False)

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
                        print('Could not run analysis for %s dim experiment on %s dataset' % (experiment, dataset))
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
    ]
    num_points_list = [
        250,
        1000,
        2000,
        4000,
        8000,
        16000,
        32000,
        64000,
        128000,
        256000,
    ]
    experiment_params = {
        'gidr_dun': {
            'optimize_method': 'gidr_dun',
            'n_neighbors': 15,
            'random_init': False,
            'umap_metric': False,
            'tsne_symmetrization': False,
            'neg_sample_rate': 1,
            'n_epochs': 500,
            'normalized': False,
            'sym_attraction': False,
            'frobenius': False,
            'angular': False,
            'tsne_scalars': True,
            'gpu': False,
            'torch': False,
            'num_threads': -1,
            'numba': True
        },
        'gidr_dun_tsne': {
            'optimize_method': 'gidr_dun',
            'n_neighbors': 15,
            'random_init': False,
            'umap_metric': False,
            'tsne_symmetrization': False,
            'neg_sample_rate': 1,
            'n_epochs': 500,
            'normalized': True,
            'sym_attraction': False,
            'frobenius': False,
            'angular': False,
            'tsne_scalars': True,
            'gpu': False,
            'torch': False,
            'num_threads': -1,
            'numba': True
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
            'torch': False,
            'num_threads': -1,
            'numba': True
        },
    }


    outputs_path = os.path.join('outputs', 'data_size_timing')
    pbar = tqdm(enumerate(datasets), total=len(datasets))
    for data_i, dataset in pbar:
        dataset_output_path = os.path.join(outputs_path, dataset)
        if not os.path.isdir(dataset_output_path):
            os.makedirs(dataset_output_path, exist_ok=True)
        for num_points in num_points_list:
            try:
                points, labels = get_dataset(dataset, num_points)
            except Exception as E:
                print('Could not find dataset %s' % dataset)
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
                        print(experiment_path)

                        instance_params = copy.copy(experiment_params[experiment])
                        instance_params['amplify_grads'] = instance_params['normalized'] # normalized and amplify_grads go together
                        if num_points > 100000:
                            instance_params['random_init'] = True
                        a, b = get_ab(instance_params['tsne_scalars'])
                        instance_params['a'] = a
                        instance_params['b'] = b
                        dr = get_algorithm('gidr_dun', instance_params, verbose=False)

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--analysis-type',
        choices=['hyper_parameter', 'data_size_sweep', 'dim_size_sweep'],
        required=True,
    )
    args = parser.parse_args()
    if args.analysis_type == 'hyper_parameter':
        cpu_analysis()
    elif args.analysis_type == 'data_size_sweep':
        data_size_timings()
    elif args.analysis_type == 'dim_size_sweep':
        dim_timings()
    else:
        raise ValueError('Unknown experiment type')
