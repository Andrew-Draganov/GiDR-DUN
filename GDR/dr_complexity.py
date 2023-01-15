import os
import time
from GDR.experiment_utils.general_utils import get_ab, make_plot
from GDR.experiment_utils.get_data import get_dataset
from GDR.experiment_utils.get_algorithm import get_algorithm
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=['mnist', 'fashion_mnist', 'cifar', 'swiss_roll', 'coil'],
        default='mnist',
        help='Which dataset to apply algorithm to'
    )
    parser.add_argument(
        '--init',
        choices=['random', 'pca', 'spectral'],
        default='spectral',
        help='How to initialize the embedding before the gradient updates'
    )
    parser.add_argument(
        '--umap-metric',
        action='store_true',
        help='If true, subtract rho\'s to calculate the umap pseudo-distance metric'
    )
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='If true, runs optimization on the GPU. Requires that GPU setup has been performed.'
    )
    parser.add_argument(
        '--num-threads',
        type=int,
        default=-1,
        help='If running CPU optimization in parallel, specifies the number of threads used'
    )

    parser.add_argument(
        '--optimize-method',
        choices=['umap', 'tsne', 'gdr'],
        default='gdr',
        help='Which embedding optimization algorithm to use'
    )
    parser.add_argument(
        '--cython',
        action='store_true',
        help="If present, run cython versions of the algorithms"
    )
    parser.add_argument(
        '--normalized',
        action='store_true',
        help='If true, normalize high- and low-dimensional pairwise likelihood matrices'
    )
    parser.add_argument(
        '--frobenius',
        action='store_true',
        help='If true, calculate gradients with respect to frobenius norm rather than KL'
    )
    parser.add_argument(
        '--num-points',
        type=int,
        default=-1,
        help='Number of samples to use from the dataset'
    )
    parser.add_argument(
        '--n-neighbors',
        type=int,
        default=15
    )
    parser.add_argument(
        '--neg-sample-rate',
        type=int,
        default=1,
        help='How many negative samples to use for each positive sample. Only applies for original UMAP'
    )
    parser.add_argument(
        '--n-epochs',
        type=int,
        default=-1,
        help='Number of times to cycle through the dataset'
    )
    args = parser.parse_args()
    return args

def fill_in_other_params(args_dict):
    """ The argparse arguments do not cover all the necessary components for get_algorithm """
    args_dict['torch'] = False
    args_dict['angular'] = False
    args_dict['sym_attraction'] = True
    args_dict['accelerated'] = False
    args_dict['tsne_symmetrization'] = True

if __name__ == '__main__':
    args = get_args()

    print('Loading %s dataset...' % args.dataset)
    points, labels = get_dataset(args.dataset, args.num_points)
    a, b = get_ab(tsne_scalars=True)
    args_dict = vars(args)
    args_dict['a'] = a
    args_dict['b'] = b
    if args_dict['n_epochs'] < 0:
        args_dict['n_epochs'] = None

    fill_in_other_params(args_dict)

    # We amplify grads in case of normalization
    args_dict['amplify_grads'] = args_dict['normalized']
    dr = get_algorithm('gdr', args_dict)

    print('fitting...')
    start = time.time()
    dr_output = dr.fit_transform(points)
    embedding = dr_output
    try:
        opt_time = dr.opt_time
    except AttributeError:
        opt_time = -1
    end = time.time()
    total_time = end - start
    print('Optimization took {:.3f} seconds'.format(opt_time))
    print('Total time took {:.3f} seconds'.format(total_time))

    save_dir = os.path.join('outputs', 'scratch')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'embedding.pdf')
    make_plot(embedding, labels, show_plot=True, save_path=save_path)
