import sys
import os
import unittest
import numpy as np

from datasets import DATASETS_PATH
from si.feature_selection.select_percentile import SelectPercentile
from si.io.csv_file import read_csv
from si.statistics.f_classification import f_classification

class TestSelectPercentile(unittest.TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        selector = SelectPercentile(percentile=50, score_func=f_classification)
        selector.fit(self.dataset)
        
        self.assertIsNotNone(selector.F)
        self.assertIsNotNone(selector.p)
        self.assertEqual(len(selector.F), self.dataset.X.shape[1])
        self.assertEqual(len(selector.p), self.dataset.X.shape[1])

    def test_transform(self):
        selector = SelectPercentile(percentile=50, score_func=f_classification)
        selector.fit(self.dataset)
        
        dataset_selected = selector.transform(self.dataset)
        
        self.assertIsNotNone(dataset_selected)
        self.assertEqual(dataset_selected.X.shape[1], 2)
        self.assertEqual(dataset_selected.X.shape[0], self.dataset.X.shape[0])
        self.assertEqual(dataset_selected.y.shape[0], self.dataset.y.shape[0])
        
        expected_features = ['petal_length', 'petal_width']
        self.assertEqual(dataset_selected.features, expected_features) 
       
        self.assertEqual(dataset_selected.label, 'class')

    def test_invalid_percentile(self):
        with self.assertRaises(ValueError):
            SelectPercentile(percentile=101)
        with self.assertRaises(ValueError):
            SelectPercentile(percentile=-1)

if __name__ == '__main__':
    unittest.main()