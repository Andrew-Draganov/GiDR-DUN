import copy
import multiprocessing
import unittest
import numpy as np
from GDR.experiment_utils.get_algorithm import get_algorithm
from GDR.tests.utils.testing_data import load_fake_data
from GDR.tests.utils.testing_params import TEST_PARAMS

class HyperParamTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = TEST_PARAMS
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
        num_cores = multiprocessing.cpu_count()
        expected_vals = {
            'a': [1.5769434, 1],
            'b': [0.8950608, 1],
            'num_threads': [num_cores, 1, 2],
            'n_epochs': [200, 10, 500],
            'n_neighbors': [2, 50],
            'neg_sample_rate': [1, 5, 30],
        }
        for switch, values in switches.items():
            for i, value in enumerate(values):
                param_test = copy.copy(self.params)
                param_test[switch] = value
                model = get_algorithm('gdr', param_test)
                if switch == 'num_threads':
                    model.set_num_threads()

                # Make sure that the model actually accepted the value correctly
                model_vars = vars(model)
                try:
                    self.assertEqual(model_vars[switch], expected_vals[switch][i])
                except AssertionError:
                    try:
                        np.testing.assert_allclose(model_vars[switch], expected_vals[switch][i])
                    except AssertionError as e:
                        raise e
                model.fit_transform(self.points)

if __name__ == '__main__':
    unittest.main()
