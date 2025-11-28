import sys
from unittest import TestCase
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from datasets import DATASETS_PATH

from si.io.csv_file import read_csv

from si.metrics.rmse import rmse
from si.models.knn_classifier import KNNClassifier

from si.model_selection.split import train_test_split
from si.models.knn_regressor import KNNRegressor

class TestKNN(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')

        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        knn = KNNClassifier(k=3)

        knn.fit(self.dataset)

        self.assertTrue(np.all(self.dataset.features == knn.dataset.features))
        self.assertTrue(np.all(self.dataset.y == knn.dataset.y))

    def test_predict(self):
        knn = KNNClassifier(k=1)

        train_dataset, test_dataset = train_test_split(self.dataset)

        knn.fit(train_dataset)
        predictions = knn.predict(test_dataset)
        self.assertEqual(predictions.shape[0], test_dataset.y.shape[0])
        self.assertTrue(np.all(predictions == test_dataset.y))

    def test_score(self):
        knn = KNNClassifier(k=3)

        train_dataset, test_dataset = train_test_split(self.dataset)

        knn.fit(train_dataset)
        score = knn.score(test_dataset)
        self.assertEqual(score, 1)

class TestKNNRegressor(TestCase):
    def setUp(self):
        """
        Prepare the testing environment by loading the dataset.
        """
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        knn = KNNRegressor(k=3)
        knn.fit(self.dataset)
        self.assertTrue(np.all(self.dataset.features == knn.dataset.features), "Features were not stored correctly.")
        self.assertTrue(np.all(self.dataset.y == knn.dataset.y), "Labels were not stored correctly.")

    def test_predict(self):
        knn = KNNRegressor(k=3) 
        
        train_dataset, test_dataset = train_test_split(self.dataset)
        knn.fit(train_dataset)
        predictions = knn.predict(test_dataset)

        self.assertEqual(predictions.shape[0], test_dataset.y.shape[0], "Prediction count does not match test samples.")

    def test_score(self):
        knn = KNNRegressor(k=3)
        train_dataset, test_dataset = train_test_split(self.dataset)
        knn.fit(train_dataset)
        predictions = knn.predict(test_dataset)
        score = knn.score(test_dataset)

        # Verify that the computed RMSE matches the expected one
        expected_score = rmse(test_dataset.y, predictions)
        self.assertAlmostEqual(score, expected_score, places=5, msg="Computed RMSE does not match the expected RMSE.")