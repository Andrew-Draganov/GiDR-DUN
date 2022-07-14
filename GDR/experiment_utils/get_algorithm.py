from GDR import GradientDR
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

def get_algorithm(algorithm_str, params, verbose=True):
    if 'gdr' == algorithm_str:
        dr = GradientDR(
                n_neighbors=params['n_neighbors'],
                n_epochs=params['n_epochs'],
                random_state=98765, # Only matters for serial-processing numba
                random_init=params['random_init'],
                pseudo_distance=params['umap_metric'],
                tsne_symmetrization=params['tsne_symmetrization'],
                optimize_method=params['optimize_method'],
                cython=params['cython'],
                torch=params['torch'],
                neg_sample_rate=params['neg_sample_rate'],
                normalized=int(params['normalized']),
                accelerated=int(params['accelerated']),
                sym_attraction=int(params['sym_attraction']),
                frob=int(params['frobenius']),
                gpu=int(params['gpu']),
                num_threads=params['num_threads'],
                angular=params['angular'],
                amplify_grads=int(params['amplify_grads']),
                a=params['a'],
                b=params['b'],
                verbose=verbose
            )
    elif algorithm_str == 'original_umap':
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError('You need to pip install umap-learn first :)')
        dr = UMAP(
            n_neighbors=params['n_neighbors'],
            n_epochs=params['n_epochs'],
            init='random' if params['random_init'] else 'spectral',
            negative_sample_rate=params['neg_sample_rate'],
            a=params['a'],
            b=params['b'],
            verbose=verbose
        )
    elif algorithm_str == 'original_tsne':
        # sklearn tsne requires at least 250 epochs
        if params['n_epochs'] is None:
            dr = TSNE(random_state=98765)
        else:
            dr = TSNE(
                random_state=98765,
                n_iter=max(250, params['n_epochs']),
                verbose=1
            )
    elif algorithm_str == 'pca':
        dr = PCA()
    elif algorithm_str == 'kernel_pca':
        dr = KernelPCA(n_components=2)
    elif algorithm_str == 'rapids_umap':
        from GDR.experiment_utils.decorator_rapids import rapids_wrapper
        dr = rapids_wrapper(
            n_neighbors=params['n_neighbors'],
            n_components=2,
            n_epochs=params['n_epochs'],
            neg_sample_rate=params['neg_sample_rate'],
            a=params['a'],
            b=params['b'],
            random_state=98765,
            umap=True,
            verbose=verbose
        )
    elif algorithm_str == 'rapids_tsne':
        from GDR.experiment_utils.decorator_rapids import rapids_wrapper
        dr = rapids_wrapper(
            n_components=2,
            verbose=verbose,
            random_state=98765,
            n_neighbors=params['n_neighbors'],
            umap=False,
        )
    else:
        raise ValueError("Unsupported algorithm")

    return dr
