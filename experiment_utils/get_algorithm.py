from gidr_dun.gidr_dun_ import UniformUmap
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from experiment_utils.decorator_rapids import rapids_tsne, rapids_umap 

def run_pymde(points):
    import pymde
    # If pyMDE, we can't use the same .fit() and .transform() methods
    # So we just have a separate case for it
    start = time.time()
    embedding = pymde.preserve_neighbors(points).embed(verbose=True)
    end = time.time()
    total_time = end - start
    return embedding, total_time

def get_algorithm(algorithm_str, params, verbose=True):
    if 'gidr_dun' in algorithm_str:
        dr = UniformUmap(
                n_neighbors=params['n_neighbors'],
                n_epochs=params['n_epochs'],
                random_state=98765, # Comment this out to turn on parallelization
                random_init=params['random_init'],
                pseudo_distance=params['umap_metric'],
                tsne_symmetrization=params['tsne_symmetrization'],
                optimize_method=params['optimize_method'],
                numba=params['numba'],
                torch=params['torch'],
                negative_sample_rate=params['neg_sample_rate'],
                normalized=int(params['normalized']),
                sym_attraction=int(params['sym_attraction']),
                frob=int(params['frobenius']),
                gpu=int(params['gpu']),
                num_threads=params['num_threads'],
                euclidean=not params['angular'],
                amplify_grads=int(params['amplify_grads']),
                a=params['a'],
                b=params['b'],
                verbose=verbose
            )
    elif algorithm_str == 'original_umap':
        dr = UMAP(
                n_neighbors=params['n_neighbors'],
                n_epochs=params['n_epochs'],
                # random_state=98765, # Comment this out to turn on parallelization
                init='random' if params['random_init'] else 'spectral',
                negative_sample_rate=params['neg_sample_rate'],
                a=params['a'],
                b=params['b'],
                verbose=verbose
            )
    elif algorithm_str == 'original_tsne':
        dr = TSNE(random_state=98765, verbose=3)
    elif algorithm_str == 'pca':
        dr = PCA()
    elif algorithm_str == 'kernel_pca':
        dr = KernelPCA(n_components=2)
    elif algorithm_str == 'rapids_umap':
        dr = rapids_umap(
            n_neighbors=params['n_neighbors'],
            n_components=2,
            n_epochs=params['n_epochs'],
            negative_sample_rate=params['neg_sample_rate'],
            a=params['a'],
            b=params['b'],
            random_state=98765,
            verbose=verbose
        )
    elif algorithm_str == 'rapids_tsne':
        dr = rapids_tsne(
            n_components=2,
            verbose=verbose,
            random_state=98765,
            n_neighbors=params['n_neighbors'])
    else:
        raise ValueError("Unsupported algorithm")

    return dr
