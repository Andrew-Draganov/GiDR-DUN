import os
import numpy as np

def read_outputs(base_path, npy_file='metrics', filter_strs=[], relevant_key=None):
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

    for directory in filtered_dirs:
        if not os.path.isfile(os.path.join(directory, npy_file)):
            continue
        print(os.path.join(directory, npy_file))
        metrics = np.load(os.path.join(directory, npy_file), allow_pickle=True)
        metrics = metrics[()]
        if relevant_key is None:
            print(directory)
            for metric, val in metrics.items():
                print('%s:' % metric, val)
            print()
        else:
            try:
                print('%s --- %s =' % (directory, relevant_key), metrics[relevant_key])
            except KeyError:
                print('WARNING --- key %s not present in %s' % (relevant_key, os.path.join(directory, npy_file)))

if __name__ == '__main__':
    # Comment out to read optimization_times for uniform_umap
    # read_outputs(
    #     'outputs',
    #     npy_file='times',
    #     filter_strs=['uniform_umap', 'dataset_size_timing'],
    #     relevant_key='opt_time'
    # )
    read_outputs(
        'outputs',
        npy_file='metrics',
        filter_strs=['cpu'],
    )
