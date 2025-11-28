import sys
import os
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.random_forest_classifier import RandomForestClassifier
from si.metrics.accuracy import accuracy

class TestRandomForestClassifier(unittest.TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit_predict_score(self):
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.3, random_state=42)

        # Create the RandomForestClassifier model
        rf = RandomForestClassifier(n_estimators=100, min_sample_split=2, max_depth=5, seed=42)
        rf.fit(train_dataset)
        self.assertEqual(len(rf.trees), 100, "The number of trained trees must equal n_estimators.")
        self.assertTrue(rf.is_fitted)
        score = rf.score(test_dataset)
        print(f"\nAccuracy on Test: {score:.4f}")
        self.assertGreater(score, 0.90)
        

        predictions = rf.predict(test_dataset)
        self.assertEqual(predictions.shape[0], test_dataset.y.shape[0])
        
        # Check if predictions are strings 
        self.assertIsInstance(predictions[0], str)
