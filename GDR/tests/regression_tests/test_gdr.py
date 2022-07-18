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
        self.num_points = int(self.points.shape[0])
        self.params = TEST_PARAMS
        self.params['num_threads'] = 1 # Random seed only works when model runs in serial
        self.values_path = os.path.join('GDR', 'tests', 'utils', 'reg_test_values')

    def check_consistency(self, embedding):
        """
        Our fake dataset has two clusters in high-dimensional space
        We expect our projection to also have two clusters in low dimensional space
        This means that intra-class distances in the embedding should be smaller than inter-class
            distances
        """
        class_inds = [np.where((self.labels - c) == 0)[0] for c in np.unique(self.labels)]
        intra_class_dists, inter_class_dists = [], []
        for i in range(int(self.num_classes * self.num_points / 2)):
            for j in range(int(self.num_classes * self.num_points / 2)):
                class_a = np.random.choice(np.arange(self.num_classes))
                if np.random.rand() > 0.5:
                    class_b = class_a
                else:
                    class_b = np.random.choice(np.arange(self.num_classes))

                index_a = np.random.choice(class_inds[class_a])
                point_a = embedding[index_a]

                index_b = np.random.choice(class_inds[class_b])
                point_b = embedding[index_b]

                dist = np.sqrt(np.sum(np.square(point_a - point_b)))
                if class_a == class_b:
                    intra_class_dists.append(dist)
                else:
                    inter_class_dists.append(dist)

        assert np.mean(intra_class_dists) < np.mean(inter_class_dists)

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
            self.check_consistency(embedding)

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
                self.check_consistency(embedding)

if __name__ == '__main__':
    unittest.main()
