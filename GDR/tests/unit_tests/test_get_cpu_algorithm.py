import unittest
from GDR.experiment_utils.get_algorithm import get_algorithm
from GDR.experiment_utils.get_data import load_fake_data

class CpuAlgTest(unittest.TestCase):
    def test_all_cpu_methods(self):
        points, _ = load_fake_data()

        params = {
            'n_neighbors': 15,
            'n_epochs': 10,
            'random_state': 98765,
            'random_init': True,
            'umap_metric': False,
            'tsne_symmetrization': False,
            'optimize_method': 'gdr',
            'cython': False,
            'torch': False,
            'neg_sample_rate': 5,
            'normalized': False,
            'sym_attraction': False,
            'frobenius': False,
            'gpu': False,
            'num_threads': -1,
            'angular': False,
            'amplify_grads': False,
            'a': None,
            'b': None,
            'verbose': True
        }

        models = [
            get_algorithm('gdr', params),
            get_algorithm('original_umap', params),
            get_algorithm('original_tsne', params),
            get_algorithm('pca', params)
        ]

if __name__ == '__main__':
    tester = CpuAlgTest()
    CpuAlgTest.test_all_cpu_methods()
