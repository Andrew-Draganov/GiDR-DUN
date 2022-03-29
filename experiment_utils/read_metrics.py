import os
import numpy as np

if __name__ == '__main__':
    datasets = [
        'mnist',
        'fashion_mnist',
        'cifar',
        'swiss_roll',
        'coil',
        'google_news',
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
    outputs_path = 'outputs'
    for data_i, dataset in enumerate(datasets):
        dataset_output_path = os.path.join(outputs_path, dataset)
        if not os.path.isdir(dataset_output_path):
            print('No dataset output path found for dataset {}'.format(dataset))
            continue

        for algorithm in algorithms:
            alg_dataset_path = os.path.join(dataset_output_path, algorithm)
            if not os.path.isdir(alg_dataset_path):
                print('No dir found at {}'.format(alg_dataset_path))
                continue

            for param in modifiable_params:
                param_path = os.path.join(alg_dataset_path, param)
                if not os.path.isdir(param_path):
                    print('No dir found at {}'.format(param_path))
                    continue

                try:
                    metrics = np.load(os.path.join(param_path, 'metrics.npy'), allow_pickle=True)
                    print('{} parameter in {} on {}'.format(param, algorithm, dataset))
                    print(metrics[()])
                    print()
                except FileNotFoundError:
                    print('Could not find metrics at {}'.format(param_path))
                    continue
