import os
import copy
import unittest
import numpy as np
from GDR.experiment_utils.get_algorithm import get_algorithm
from GDR.tests.utils.testing_data import load_fake_clusters
from GDR.tests.utils.testing_params import TEST_PARAMS
from sklearn.cluster import KMeans

class GdrTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = 2
        self.points, self.labels = load_fake_clusters(num_classes=self.num_classes)
        self.params = TEST_PARAMS
        self.params['num_threads'] = 1 # Random seed only works when model runs in serial
        self.values_path = os.path.join('GDR', 'tests', 'utils', 'reg_test_values')

    def test_bool_hyperparams(self):
        """
        Assert that we get the expected embedding when toggling each of the below hyperparameters
        """
        switches = [
            'random_init',
            'umap_metric',
            'tsne_symmetrization',
            'normalized',
            'accelerated',
            'sym_attraction',
            'frobenius',
            'amplify_grads'
        ]
        results = {}
        for switch in switches:
            param_test = copy.copy(self.params)
            param_test[switch] = not param_test[switch]
            model = get_algorithm('gdr', param_test)
            embedding = model.fit_transform(self.points)
            results[switch] = embedding

        bool_vals_path = os.path.join(self.values_path, 'bool_reg_values.npy')
        np.save(bool_vals_path, results, allow_pickle=True)
        correct_values = np.load(bool_vals_path, allow_pickle=True)[()]
        for switch, correct_embedding in correct_values.items():
            np.testing.assert_allclose(correct_embedding, results[switch])

    def test_valued_hyperparams(self):
        """
        Assert that we get the expected embedding for each of the following non-boolean
        hyperparameter values
        """
        switches = {
            'a': [None, 1],
            'b': [None, 1],
            'n_epochs': [None, 10, 500],
            'n_neighbors': [2, 50],
            'neg_sample_rate': [1, 5, 30],
        }
        results = {switch: [] for switch in switches}
        for switch, values in switches.items():
            for i, value in enumerate(values):
                param_test = copy.copy(self.params)
                param_test[switch] = value
                model = get_algorithm('gdr', param_test)
                embedding = model.fit_transform(self.points)
                results[switch].append(embedding)

        non_bool_vals_path = os.path.join(self.values_path, 'non_bool_reg_values.npy')
        correct_values = np.load(non_bool_vals_path, allow_pickle=True)[()]
        for switch, values in correct_values.items():
            for i, value in enumerate(values):
                np.testing.assert_allclose(value, results[switch][i])


if __name__ == '__main__':
    unittest.main()
