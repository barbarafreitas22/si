import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from si.io.csv_file import read_csv
from si.models.logistic_regression import LogisticRegression
from si.model_selection.randomized_search import randomized_search_cv

def test_randomized_search_breast_bin():
    datasets_base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')
    filename = os.path.join(datasets_base_path, 'breast_bin', 'breast-bin.csv')
    dataset = read_csv(filename, sep=",", features=True, label=True)

    model = LogisticRegression()

    l2_penalty_dist = np.linspace(1, 10, 10)
    alpha_dist = np.linspace(0.001, 0.0001, 100)
    max_iter_dist = np.linspace(1000, 2000, 200, dtype=int)

    hyperparameter_grid = {
        'l2_penalty': l2_penalty_dist,
        'alpha': alpha_dist,
        'max_iter': max_iter_dist
    }

    results = randomized_search_cv(
        model=model,
        dataset=dataset,
        hyperparameter_grid=hyperparameter_grid,
        cv=3,
        n_iter=10,
        random_state=42
    )

    print(f"Best Score: {results['best_score']}")
    print("Best Hyperparameters:")
    for param, value in results['best_hyperparameters'].items():
        print(f"{param}: {value}")

    print("All scores:")
    for i, score in enumerate(results['scores']):
        print(f"Iteration {i+1}: {score} {results['hyperparameters'][i]}")

if __name__ == '__main__':
    test_randomized_search_breast_bin()