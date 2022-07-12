import os
import copy
import unittest
import numpy as np
from GDR.optimizer.graph_weights import \
    get_sigmas_and_rhos, \
    get_similarities
from GDR.experiment_utils.get_algorithm import get_algorithm
from GDR.tests.utils.testing_data import load_fake_data
from GDR.tests.utils.testing_params import TEST_PARAMS

class HyperParamTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.points, _ = load_fake_data()
        self.model = get_algorithm('gdr', TEST_PARAMS)
        self.model.get_nearest_neighbors(self.points)

        self.dists = self.model._knn_dists
        self.inds = self.model._knn_indices.astype(np.int64)
        self.sigmas, self.rhos = get_sigmas_and_rhos(self.dists, 15)
        self.values_path = os.path.join('GDR', 'tests', 'utils', 'unit_test_values')

    def test_neighbors(self):
        knn_dists = self.model._knn_dists
        knn_indices = self.model._knn_indices

        neighbor_val_path = os.path.join(self.values_path, 'knn.npy')
        correct_values = np.load(neighbor_val_path, allow_pickle=True)[()]
        np.testing.assert_allclose(knn_dists, correct_values['dists'])
        np.testing.assert_allclose(knn_indices, correct_values['inds'])

    def test_sigmas_rhos(self):
        sigma_rho_path = os.path.join(self.values_path, 'sigmas_rhos.npy')
        correct_values = np.load(sigma_rho_path, allow_pickle=True)[()]
        np.testing.assert_allclose(self.sigmas, correct_values['sigmas'])
        np.testing.assert_allclose(self.rhos, correct_values['rhos'])

    def test_pseudo_dist_similarities(self):
        rows, cols, vals = get_similarities(
            self.inds,
            self.dists,
            self.sigmas,
            self.rhos,
            pseudo_distance=True
        )
        sim_value_path = os.path.join(self.values_path, 'pseudo_dist_similarities.npy')
        correct_values = np.load(sim_value_path, allow_pickle=True)[()]
        np.testing.assert_allclose(rows, correct_values['rows'])
        np.testing.assert_allclose(cols, correct_values['cols'])
        np.testing.assert_allclose(vals, correct_values['vals'])

    def test_normal_dist_similarities(self):
        rows, cols, vals = get_similarities(
            self.inds,
            self.dists,
            self.sigmas,
            self.rhos,
            pseudo_distance=False
        )
        sim_value_path = os.path.join(self.values_path, 'normal_dist_similarities.npy')
        correct_values = np.load(sim_value_path, allow_pickle=True)[()]
        np.testing.assert_allclose(rows, correct_values['rows'])
        np.testing.assert_allclose(cols, correct_values['cols'])
        np.testing.assert_allclose(vals, correct_values['vals'])
