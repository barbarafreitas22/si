from unittest import TestCase
import os
import numpy as np

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.model_selection.split import stratified_train_test_split

class TestStratifiedSplit(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_stratified_split(self):
        # 20% for testing (test_size = 0.2)
        test_size = 0.2
        train, test = stratified_train_test_split(self.dataset, test_size=test_size, random_state=42)
        # Check Sizes
        self.assertEqual(train.shape()[0] + test.shape()[0], self.dataset.shape()[0])

        # Check Stratification, the Iris dataset has 3 classes with 50 samples each.
        
        classes = self.dataset.get_classes()
        
        for c in classes:
            # Count how many times class c appears in test and train sets
            count_test = np.sum(test.y == c)
            count_train = np.sum(train.y == c)
            self.assertEqual(count_test, 10, f"Class {c} should have 10 samples in test, but has {count_test}")
            self.assertEqual(count_train, 40, f"Class {c} should have 40 samples in train, but has {count_train}")