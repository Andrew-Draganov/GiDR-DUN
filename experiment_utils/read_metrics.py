import os
import numpy as np
import matplotlib.pyplot as plt

def make_bar_plots(outputs):
    all_opt_methods = {}
    for metric, datasets in outputs.items():
        num_datasets = len(datasets)
        for i, (dataset, opt_methods) in enumerate(datasets.items()):
            if dataset == 'swiss_roll':
                continue
            for opt_method in opt_methods:
                if opt_method not in all_opt_methods:
                    all_opt_methods[opt_method] = {}
                all_opt_methods[opt_method][dataset] = {}
                param_vals = np.array(list(opt_methods[opt_method].values()))
                all_opt_methods[opt_method][dataset]['mean'] = np.mean(param_vals)
                all_opt_methods[opt_method][dataset]['dev'] = np.std(param_vals)


    # We don't run tsne on google news, so fill it with 0's
    all_opt_methods['tsne']['google_news'] = {}
    all_opt_methods['tsne']['google_news']['mean'] = 0
    all_opt_methods['tsne']['google_news']['dev'] = 0
    dataset_loc_dict = {
        'mnist': 2,
        'fashion_mnist': 3,
        'cifar': 4,
        'coil': 5,
        'google_news': 6
    }
    colors = {
        'tsne': 'red',
        'umap': 'blue',
        'uniform_umap': 'green',
        'uniform_umap_tsne': 'purple',
    }
    method_labels = {
        'tsne': 'TSNE',
        'umap': 'UMAP',
        'uniform_umap': 'GDR-umap',
        'uniform_umap_tsne': 'GDR-tsne'
    }
    num_opt_methods = len(all_opt_methods)
    for i, opt_method in enumerate(all_opt_methods):
        datasets = all_opt_methods[opt_method]
        plt.bar(
            [dataset_loc_dict[dataset] * (num_opt_methods+1) + 1 - (num_opt_methods+1)/2 + i for j, dataset in enumerate(datasets)],
            [all_opt_methods[opt_method][dataset]['mean'] for dataset in all_opt_methods[opt_method]],
            yerr=[all_opt_methods[opt_method][dataset]['dev'] for dataset in all_opt_methods[opt_method]],
            width=1,
            color=colors[opt_method]
        )
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[opt_method]) for opt_method in all_opt_methods]
    labels = [method_labels[opt_method] for opt_method in all_opt_methods]
    plt.legend(handles, labels)
    plt.ylabel('Runtime in seconds')
    plt.xticks(
        [i * (num_opt_methods + 1) + 2.5 - (num_opt_methods+1)/2 for i in dataset_loc_dict.values()],
        list(dataset_loc_dict.keys())
    )
    plt.show()

def print_row_means(outputs):
    for metric, metric_dict in outputs.items():
        print(metric)
        for dataset, dataset_dict in metric_dict.items():
            print(dataset)
            for opt_method, params_dict in dataset_dict.items():
                print(opt_method)
                params_dict.pop('neg_sample_rate')
                params_dict.pop('normalized')
                param_outputs = np.array(list(params_dict.values()))
                print('{} metric mean on {} dataset with {}: {:.4f}'.format(metric, dataset, opt_method, np.mean(param_outputs)))
                print('{} metric std. dev. on {} dataset with {}: {:.4f}'.format(metric, dataset, opt_method, np.std(param_outputs)))
                print()
            print()
            print()
        print()
        print()
        print()


def print_col_means(outputs):
    # Subtract row-means
    column_means = {}
    for metric, metric_dict in outputs.items():
        if metric not in column_means:
            column_means[metric] = {}
        for dataset, dataset_dict in metric_dict.items():
            if dataset not in column_means[metric]:
                column_means[metric][dataset] = {}
            for opt_method, params_dict in dataset_dict.items():
                params_dict.pop('neg_sample_rate')
                params_dict.pop('normalized')
                param_outputs = np.array(list(params_dict.values()))
                for param, value in params_dict.items():
                    if param not in column_means[metric]:
                        column_means[metric][dataset][param] = []
                    column_means[metric][dataset][param].append(value - np.mean(param_outputs))

    for metric, metric_dict in column_means.items():
        print(metric)
        for dataset, dataset_dict in metric_dict.items():
            print(dataset)
            for param, value_list in dataset_dict.items():
                print(metric, dataset, param, np.mean(value_list))
            print()


def read_outputs(
    base_path,
    print_outputs=True,
    npy_file='metrics',
    filter_strs=[],
    relevant_key=None
):
    if not npy_file.endswith('.npy'):
        npy_file += '.npy'

    subdirs = [d[0] for d in os.walk(base_path)]
    if filter_strs:
        filtered_dirs = subdirs
        for filter_str in filter_strs:
            temp_filtered = []
            for sd in filtered_dirs:
                if filter_str in sd:
                    temp_filtered += [sd]
            filtered_dirs = temp_filtered
    else:
        filtered_dirs = subdirs

    if not filtered_dirs:
        raise ValueError("No output directories available after filtering")

    gathered_outputs = {}
    for directory in filtered_dirs:
        # Get path for results and corresponding metadata
        current_path = os.path.join(directory, npy_file)
        if not os.path.isfile(current_path):
            continue
        print(current_path)
        subsets = current_path.split('/')
        dataset = subsets[2]
        opt_method = subsets[3]
        param = subsets[4]

        # Load results from .npy file
        metrics = np.load(current_path, allow_pickle=True)
        metrics = metrics[()]

        if relevant_key is None:
            # If we don't care about one specific metric, loop through all of them
            for metric, val in metrics.items():
                # Make sure dicts are formatted according to all metadata
                if metric not in gathered_outputs:
                    gathered_outputs[metric] = {}
                if dataset not in gathered_outputs[metric]:
                    gathered_outputs[metric][dataset] = {}
                if opt_method not in gathered_outputs[metric][dataset]:
                    gathered_outputs[metric][dataset][opt_method] = {}
                if param not in gathered_outputs[metric][dataset][opt_method]:
                    gathered_outputs[metric][dataset][opt_method][param] = {}
                gathered_outputs[metric][dataset][opt_method][param] = val

                if print_outputs:
                    print('%s:' % metric, val)
            if print_outputs:
                print()
        else:
            try:
                # Same as above but for just one specific metric that we care about
                if relevant_key not in gathered_outputs:
                    gathered_outputs[relevant_key] = {}
                if dataset not in gathered_outputs[relevant_key]:
                    gathered_outputs[relevant_key][dataset] = {}
                if opt_method not in gathered_outputs[relevant_key][dataset]:
                    gathered_outputs[relevant_key][dataset][opt_method] = {}
                if param not in gathered_outputs[relevant_key][dataset][opt_method]:
                    gathered_outputs[relevant_key][dataset][opt_method][param] = {}
                gathered_outputs[relevant_key][dataset][opt_method][param] = metrics[relevant_key]

                if print_outputs:
                    print(
                        '%s --- %s =' % (directory, relevant_key),
                        '{:.4f}'.format(metrics[relevant_key])
                    )
                    print()
            except KeyError:
                print('WARNING --- key %s not present in %s' % (relevant_key, current_path))

    return gathered_outputs

if __name__ == '__main__':
    # Comment out to make bar-plots of optimization_times for cpu
    cpu_times = read_outputs(
        'outputs',
        npy_file='times',
        filter_strs=['cpu'],
        relevant_key='total_time'
    )
    make_bar_plots(cpu_times)


    # outputs = read_outputs(
    #     'outputs',
    #     npy_file='metrics',
    #     filter_strs=['cpu'],
    # )
    # print_row_means(outputs)
    # print_col_means(outputs)
