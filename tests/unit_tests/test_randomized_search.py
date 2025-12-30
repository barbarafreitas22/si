from unittest import TestCase
import os
import numpy as np

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.models.logistic_regression import LogisticRegression
from si.model_selection.randomized_search import randomized_search_cv

class TestRandomizedSearch(TestCase):

    def test_randomized_search_cv_protocol(self):
        csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        dataset = read_csv(filename=csv_file, label=True, sep=",")
        model = LogisticRegression()

        # Grid with the asked hyperparameter distributions
        hyperparameter_grid = {
            'l2_penalty': np.linspace(1, 10, 10),
            'alpha': np.linspace(0.001, 0.0001, 100),
            'max_iter': np.linspace(1000, 2000, 200).astype(int)
        }

        # Run Randomized Search (n_iter=10, cv=3)
        results = randomized_search_cv(
            model=model,
            dataset=dataset,
            hyperparameter_grid=hyperparameter_grid,
            cv=3,
            n_iter=10
        )

        # Results
        print(f"Best score: {results['best_score']:.4f}")
        print(f"Best hyperparameters: {results['best_hyperparameters']}")

        # Asserts
        self.assertEqual(len(results['scores']), 10)
        self.assertEqual(len(results['hyperparameters']), 10)
        self.assertTrue(0 <= results['best_score'] <= 1)