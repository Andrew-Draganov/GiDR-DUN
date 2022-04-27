from uniform_umap.uniform_umap_ import UniformUmap
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from experiment_utils.decorator_rapids import * 

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
    if 'uniform_umap' in algorithm_str:
        dr = UniformUmap(
                n_neighbors=params['n_neighbors'],
                n_epochs=params['n_epochs'],
                random_state=12345, # Comment this out to turn on parallelization
                random_init=params['random_init'],
                pseudo_distance=params['umap_metric'],
                tsne_symmetrization=params['tsne_symmetrization'],
                optimize_method=params['optimize_method'],
                numba=params['numba'],
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
    elif algorithm_str == 'original_tsne':
        dr = TSNE(random_state=12345, verbose=3)
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
            random_state=12345,
            # verbose=True
            verbose=verbose
        )
    elif algorithm_str == 'rapids_tsne':
        dr = rapids_tsne(
            n_components=2,
            learning_rate=params['learning_rate'],
            n_epochs=params['n_epochs'], 
            # verbose=True,
            verbose=verbose,
            random_state=12345,
            n_neighbors=params['n_neighbors'])
    elif algorithm_str == 'rapids_umap_org':
        n_neighbors=params['n_neighbors']
        n_components=2
        n_epochs=params['n_epochs']
        negative_sample_rate=params['neg_sample_rate']
        a=params['a']
        b=params['b']
        random_state=12345
        # verbose=True
        verbose=verbose

        dr = cumlUMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            n_epochs=n_epochs,
            # learning_rate=params['learning_rate'], #already default to 1.0 in rapids
            #random_init=params['random_init'], #default spectral
            #min_dist=params['min_dist'], #default 0.1
            #spread=params['spread'], #default 1.0
            #set_op_mix_ratio=params['set_op_mix_ratio'], #default 1.0
            #local_connectivity=params['local_connectivity'], #default 1
            #repulsion_strength=params['repulsion_strength'], #default 1.0
            negative_sample_rate=negative_sample_rate,
            #transform_queue_size=params['transform_queue_size'], #default 4
            a=a,
            b=b,
            #hash_input=['hash_input'], #default false
            random_state=random_state,
            verbose=verbose,
            output_type='numpy'
        )
    elif algorithm_str == 'rapids_tsne_org':
        n_components=2
        learning_rate=params['learning_rate']
        n_epochs=params['n_epochs']
        # verbose=True
        verbose=verbose
        random_state=12345
        n_neighbors=params['n_neighbors']

        dr = cumlTSNE(
            n_components=n_components,
            #perplexity=params['perplexity'], #default 30.0
            #early_exaggeration=['early_exaggeration'], #default 12.0
            #late_exaggeration=['late_exaggeration'], #default 1.0
            learning_rate=learning_rate, #default 200.0
            n_iter=n_epochs, #default 1000
            #n_iter_without_progress seems to not be used
            # min_grad_norm default 1e-07
            # metricstr ‘euclidean’ only (default ‘euclidean’)
            #initstr ‘random’ (default ‘random’)
            verbose=verbose,
            random_state=random_state,
            method = 'barnes_hut', #method str ‘barnes_hut’, ‘fft’ or ‘exact’ (default ‘barnes_hut’)
            #anglefloat (default 0.5)
            learning_rate_method = None,#learning_rate_method str ‘adaptive’, ‘none’ or None (default ‘adaptive’)
            n_neighbors=n_neighbors, #(default 90)
            #perplexity_max_iterint (default 100)
            #exaggeration_iterint (default 250)
            #pre_momentumfloat (default 0.5)
            #post_momentumfloat (default 0.8)
            #square_distancesboolean, default=True
            output_type='numpy'
        )
    else:
        raise ValueError("Unsupported algorithm")

    return dr
