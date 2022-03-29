from uniform_umap.uniform_umap_ import UniformUmap
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

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
    if algorithm_str == 'uniform_umap':
        dr = UniformUmap(
                n_neighbors=params['n_neighbors'],
                n_epochs=params['n_epochs'],
                random_state=12345, # Comment this out to turn on parallelization
                random_init=params['random_init'],
                pseudo_distance=params['umap_metric'],
                tsne_symmetrization=params['tsne_symmetrization'],
                optimize_method=params['optimize_method'],
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
                # random_state=12345, # Comment this out to turn on parallelization
                init='random' if params['random_init'] else 'spectral',
                negative_sample_rate=params['neg_sample_rate'],
                a=params['a'],
                b=params['b'],
                verbose=verbose
            )
    elif algorithm_str == 'tsne':
        dr = TSNE(random_state=12345, verbose=3)
    elif algorithm_str == 'pca':
        dr = PCA()
    elif algorithm_str == 'kernel_pca':
        dr = KernelPCA(n_components=2)
    else:
        raise ValueError("Unsupported algorithm")

    return dr
