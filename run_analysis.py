import os
import copy
import time
from tqdm import tqdm
import numpy as np
from utils.metrics import classifier_accuracy, cluster_quality, cluster_distances
from utils.general_utils import get_ab, make_plot
from utils.get_data import get_dataset
from utils.get_algorithm import get_algorithm

if __name__ == '__main__':
    datasets = [
        'mnist',
        'fashion_mnist',
        'cifar',
        'swiss_roll',
        'coil',
        'google_news',
    ]
    num_points = [
        60000,
        60000,
        50000,
        5000,
        7200,
        350000,
    ]
    algorithms = [
        'umap',
        'tsne',
        'uniform_umap'
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
        },
        # FIXME -- umap with neg-sample-rate = 1 
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
            'tsne_scalars': False,
            'gpu': False,
            'num_threads': -1,
        }
    }

    outputs_path = 'outputs'
    pbar = tqdm(enumerate(datasets), total=len(datasets))
    for data_i, dataset in pbar:
        points, labels = get_dataset(dataset, num_points[data_i])
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
                    embedding, opt_time = dr.fit_transform(points)
                    end = time.time()
                    total_time = end - start

                    make_plot(
                        embedding,
                        labels,
                        save_path=os.path.join(param_path, 'embedding.png')
                    )
                    metrics = {
                        # 'cluster_dists': cluster_distances(embedding, labels),
                        'knn_accuracy': classifier_accuracy(embedding, labels, 100),
                        'v-score': cluster_quality(embedding, labels)
                    }
                    times = {
                        'opt_time': opt_time,
                        'total_time': total_time
                    }
                    np.save(os.path.join(param_path, "metrics.npy"), metrics)
                    np.save(os.path.join(param_path, "times.npy"), times)
                    np.save(os.path.join(param_path, "embedding.npy"), embedding)
                    np.save(os.path.join(param_path, "labels.npy"), labels)
                except:
                    continue
