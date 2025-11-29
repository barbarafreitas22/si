from unittest import TestCase
import numpy as np
import os

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.models.knn_regressor import KNNRegressor
from si.model_selection.split import train_test_split

class TestKNNRegressor(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        knn = KNNRegressor(k=3)
        
        knn.fit(self.dataset)
        # Checks if the dataset was stored correctly
        self.assertTrue(np.all(self.dataset.features == knn.dataset.features))
        self.assertTrue(np.all(self.dataset.y == knn.dataset.y))

    def test_predict(self):
        knn = KNNRegressor(k=3)

        train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.2)

        knn.fit(train_dataset)
        predictions = knn.predict(test_dataset)
        self.assertEqual(predictions.shape[0], test_dataset.y.shape[0])
        self.assertTrue(np.issubdtype(predictions.dtype, np.floating))

    
    def test_score(self):
        # Normal test with the CPU dataset
        knn = KNNRegressor(k=3)
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.2)
        
        knn.fit(train_dataset)
        score = knn.score(test_dataset)
        
        # The score is RMSE
        self.assertIsInstance(score, float)
        self.assertTrue(score >= 0)

        # Test overfitting scenario, creates a small dataset where we know there are no duplicates
        from si.data.dataset import Dataset
        X_dummy = np.array([[1, 1], [2, 2], [3, 3]])
        y_dummy = np.array([10, 20, 30])
        dummy_dataset = Dataset(X=X_dummy, y=y_dummy)

        knn_perfect = KNNRegressor(k=1)
        knn_perfect.fit(dummy_dataset)
        
        # The error must be 0.0
        dummy_score = knn_perfect.score(dummy_dataset)
        self.assertAlmostEqual(dummy_score, 0.0, places=5)