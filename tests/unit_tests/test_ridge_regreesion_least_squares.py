import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares
from si.metrics.mse import mse

class TestRidgeRegressionLeastSquares(unittest.TestCase):
    """
    Unit tests for the RidgeRegressionLeastSquares class.
    """

    def setUp(self):
        """
        Prepares the test environment by loading the regression dataset (CPU).
        """
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        model = RidgeRegressionLeastSquares(l2_penalty=0.5, scale=True)
        model.fit(self.dataset)
        self.assertIsNotNone(model.theta)
        self.assertIsNotNone(model.theta_zero)
        self.assertEqual(len(model.theta), self.dataset.X.shape[1])
        self.assertIsNotNone(model.mean)
        self.assertIsNotNone(model.std)

    def test_predict(self):
        """
        Tests if the predict method generates predictions with the correct shape.
        """
        model = RidgeRegressionLeastSquares(l2_penalty=0.5, scale=True)
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.3, random_state=42)
        
        model.fit(train_dataset)
        predictions = model.predict(test_dataset)
        self.assertEqual(predictions.shape[0], test_dataset.y.shape[0])
        self.assertFalse(np.isnan(predictions).any())

    def test_score(self):
        model = RidgeRegressionLeastSquares(l2_penalty=0.5, scale=True)
        
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.3, random_state=42)
        
        model.fit(train_dataset)
        model_score = model.score(test_dataset)
        predictions = model.predict(test_dataset)
        expected_score = mse(test_dataset.y, predictions)
        self.assertAlmostEqual(model_score, expected_score, places=5)
        self.assertGreaterEqual(model_score, 0.0)
