import re
import os
import numpy as np
import matplotlib.pyplot as plt

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



def make_gpu_bar_plots(outputs, plot_save_dir=''):
    all_opt_methods = {
        'rapids_tsne': {},
        'rapids_umap': {},
        'recreate_tsne_gpu': {},
        'recreate_umap_gpu': {}
    }
    for metric, datasets in outputs.items():
        num_datasets = len(datasets)
        for i, (dataset, opt_methods) in enumerate(datasets.items()):
            if dataset == 'swiss_roll':
                continue
            for opt_method in opt_methods:
                if opt_method not in all_opt_methods:
                    all_opt_methods[opt_method] = {}
                param_vals = np.array(list(opt_methods[opt_method].values()))
                all_opt_methods[opt_method][dataset] = np.mean(param_vals)


    # Rapids runs out of memory on COIL-100 dataset
    all_opt_methods['rapids_tsne']['coil'] = 0
    all_opt_methods['rapids_umap']['coil'] = 0
    dataset_loc_dict = {
        'coil': 2,
        'mnist': 3,
        'fashion_mnist': 4,
        'google_news': 5
    }
    colors = {
        'rapids_tsne': 'red',
        'rapids_umap': 'blue',
        'recreate_umap_gpu': 'green',
        'recreate_tsne_gpu': 'purple',
    }
    method_labels = {
        'rapids_tsne': 'Rapids TSNE',
        'rapids_umap': 'Rapids UMAP',
        'recreate_umap_gpu': 'GDR-umap',
        'recreate_tsne_gpu': 'GDR-tsne'
    }
    num_opt_methods = len(all_opt_methods)
    for i, opt_method in enumerate(all_opt_methods):
        datasets = all_opt_methods[opt_method]
        plt.bar(
            [dataset_loc_dict[dataset] * (num_opt_methods+1) + 1 - (num_opt_methods+1)/2 + i for j, dataset in enumerate(datasets)],
            [all_opt_methods[opt_method][dataset] for dataset in all_opt_methods[opt_method]],
            width=1,
            color=colors[opt_method]
        )
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[opt_method]) for opt_method in all_opt_methods]
    labels = [method_labels[opt_method] for opt_method in all_opt_methods]
    plt.legend(handles, labels)
    plt.ylabel('Runtime in seconds')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.xticks(
        [i * (num_opt_methods + 1) + 2.5 - (num_opt_methods+1)/2 for i in dataset_loc_dict.values()],
        list(dataset_loc_dict.keys())
    )
    plt.savefig(os.path.join(plot_save_dir, 'gpu_bar_plot.png'))
    plt.close()


def make_line_plots(outputs, x_axis_title='', plot_save_dir=''):
    all_opt_methods = {
        'rapids_tsne': {},
        'rapids_umap': {},
        'recreate_tsne_gpu': {},
        'recreate_umap_gpu': {}
    }
    all_sweep_params = []
    for metric, datasets in outputs.items():
        for i, (dataset, dataset_dict) in enumerate(datasets.items()):
            for dim_str, dim_dict in dataset_dict.items():
                dim = int(re.search(r'\d+', dim_str).group())
                if dim not in all_sweep_params:
                    all_sweep_params.append(dim)
                for opt_method in dim_dict:
                    if opt_method not in all_opt_methods:
                        all_opt_methods[opt_method] = {}
                    all_opt_methods[opt_method][dim] = dim_dict[opt_method]
    all_sweep_params = sorted(all_sweep_params)

    colors = {
        'rapids_tsne': 'red',
        'rapids_umap': 'blue',
        'recreate_umap_gpu': 'green',
        'recreate_tsne_gpu': 'purple',
    }
    method_labels = {
        'rapids_tsne': 'Rapids TSNE',
        'rapids_umap': 'Rapids UMAP',
        'recreate_umap_gpu': 'GDR-umap',
        'recreate_tsne_gpu': 'GDR-tsne'
    }
    markers = {
        'rapids_tsne': '+',
        'rapids_umap': 'o',
        'recreate_umap_gpu': '.',
        'recreate_tsne_gpu': 'X'
    }
    num_opt_methods = len(all_opt_methods)
    for i, (opt_method, times_dict) in enumerate(all_opt_methods.items()):
        opt_sweep_params = sorted(list(times_dict.keys()))
        opt_method_times = [times_dict[d] for d in opt_sweep_params]
        plt.plot(
            range(len(opt_sweep_params)),
            opt_method_times,
            color=colors[opt_method],
            marker=markers[opt_method]
        )
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[opt_method]) for opt_method in all_opt_methods]
    labels = [method_labels[opt_method] for opt_method in all_opt_methods]
    plt.legend(handles, labels)
    plt.ylabel('Runtime in seconds')
    plt.xlabel(x_axis_title)
    ax = plt.gca()
    ax.set_yscale('log')
    plt.xticks(range(len(all_sweep_params)), all_sweep_params)
    plt.savefig(os.path.join(plot_save_dir, 'gpu_line_plot_{}.png'.format(x_axis_title)))
    plt.show()
    plt.close()



if __name__ == '__main__':
    plot_save_dir = os.path.join('outputs/gpu/figures')
    os.makedirs(plot_save_dir, exist_ok=True)
    gpu_times = read_outputs(
        'outputs/gpu',
        npy_file='times',
        filter_strs=['gpu'],
        relevant_key='total_time'
    )
    make_gpu_bar_plots(gpu_times, plot_save_dir)

    dim_times = read_outputs(
        'outputs/dim_timing/mnist',
        npy_file='times',
        relevant_key='total_time'
    )
    make_line_plots(dim_times, 'X dimensionality', plot_save_dir)

    data_size_times = read_outputs(
        'outputs/data_size_timing/mnist',
        npy_file='times',
        relevant_key='total_time'
    )
    make_line_plots(data_size_times, 'Number of points', plot_save_dir)

