import copy
import unittest
from GDR.experiment_utils.get_algorithm import get_algorithm
from GDR.experiment_utils.get_data import load_fake_data

class HyperParamTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = {
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
        self.points, _ = load_fake_data()

    def test_cpu_methods(self):
        models = [
            get_algorithm('gdr', self.params),
            get_algorithm('original_umap', self.params),
            get_algorithm('original_tsne', self.params),
            get_algorithm('pca', self.params)
        ]
        for model in models:
            model.fit_transform(self.points)

    def test_bool_hyperparams(self):
        switches = [
            'random_init',
            'umap_metric',
            'tsne_symmetrization',
            'normalized',
            'sym_attraction',
            'frobenius',
            'amplify_grads'
        ]
        for switch in switches:
            param_test = copy.copy(self.params)
            param_test[switch] = not param_test[switch]
            model = get_algorithm('gdr', param_test)
            model.fit_transform(self.points)

    def test_valued_hyperparams(self):
        switches = {
            'a': [None, 1],
            'b': [None, 1],
            'num_threads': [-1, 1, 2],
            'n_epochs': [None, 10, 500],
            'n_neighbors': [2, 50],
            'neg_sample_rate': [1, 5, 30],
            }
        for switch, values in switches.items():
            for value in values:
                param_test = copy.copy(self.params)
                param_test[switch] = value
                model = get_algorithm('gdr', param_test)
                model.fit_transform(self.points)

if __name__ == '__main__':
    unittest.main()
