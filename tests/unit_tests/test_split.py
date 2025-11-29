import sys
import os
from unittest import TestCase
import numpy as np

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.model_selection.split import stratified_train_test_split, train_test_split

class TestSplits(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_train_test_split(self):
        train, test = train_test_split(self.dataset, test_size=0.2, random_state=123)
        test_samples_size = int(self.dataset.shape()[0] * 0.2)
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)

    def test_stratified_train_test_split(self):
        test_size = 0.3
        train, test = stratified_train_test_split(self.dataset, test_size=test_size, random_state=42)
        
        test_samples_size = int(self.dataset.shape()[0] * test_size)
        
        self.assertEqual(test.shape()[0], test_samples_size)
        self.assertEqual(train.shape()[0], self.dataset.shape()[0] - test_samples_size)
        
        test_classes, counts = np.unique(test.y, return_counts=True)
        self.assertTrue(np.all(counts == 15))