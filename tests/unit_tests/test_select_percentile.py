import unittest
import numpy as np
from si.data.dataset import Dataset
from si.feature_selection.select_percentile import SelectPercentile

# Mock function to test specific F-score scenarios without external dependencies
def mock_score_func_tie_breaker(dataset):
    """
    Returns fixed F-scores to test the tie-breaking logic.
    Scenario from Exercise: 10 features, 40% selection (k=4).
    F-scores: [1.2, 3.4, 2.1, 5.6, 4.3, 5.6, 7.8, 6.5, 5.6, 3.2]
    Ties at 5.6 are at indices 3, 5, 8. The stable sort must pick 3 and 5.
    """
    F_scores = np.array([1.2, 3.4, 2.1, 5.6, 4.3, 5.6, 7.8, 6.5, 5.6, 3.2])
    p_values = np.zeros_like(F_scores) 
    return F_scores, p_values


class TestSelectPercentile(unittest.TestCase):

    def test_selection_percentage_basic(self):
        """Tests basic selection: 50% of 4 features should select 2."""
        
        X = np.random.rand(10, 4) 
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        features = ['f1', 'f2', 'f3', 'f4']
        dataset = Dataset(X, y, features=features, label="y")

        # Initialize selector to keep 50%
        selector = SelectPercentile(percentile=50) 
    
        selector.fit(dataset)
        new_dataset = selector.transform(dataset)
        self.assertEqual(new_dataset.shape(), (10, 2))
        self.assertEqual(new_dataset.X.shape[1], 2)
        
    def test_tie_breaking_logic(self):
        """
        Tests the requirement that exactly k features are selected,
        using the stable sort to resolve ties by original index order.
        """
        
        X = np.random.rand(5, 10) 
        y = np.array([0, 1, 0, 1, 0])
        # Feature names f_0 to f_9
        features = [f"f_{i}" for i in range(10)] 
        dataset_tie = Dataset(X, y, features=features, label="y")

        # Initialize selector with mock function and 40% (k=4)
        selector = SelectPercentile(score_func=mock_score_func_tie_breaker, percentile=40)
        selector.fit(dataset_tie)
        
        new_dataset = selector.transform(dataset_tie)

        # 1. Assert exactly 4 features were selected (40% of 10)
        self.assertEqual(len(new_dataset.features), 4)
        
        # 2. Assert the correct features were selected due to stable sort:
        # Top scores: f_6 (7.8), f_7 (6.5)
        # Ties (5.6): f_3, f_5, f_8. Stable sort ensures f_3 and f_5 are picked.
        # Expected indices (sorted for output): 3, 5, 6, 7
        expected_features = ['f_3', 'f_5', 'f_6', 'f_7']
        
        self.assertListEqual(new_dataset.features, expected_features)
        
    def test_selection_percentage_zero(self):
        """Tests if 0% selection returns zero features."""
        X = np.random.rand(5, 5) 
        y = np.array([0, 1, 0, 1, 0])
        dataset = Dataset(X, y)
        
        selector = SelectPercentile(percentile=0)
        selector.fit(dataset)
        new_dataset = selector.transform(dataset)

        # Check for 0 features
        self.assertEqual(new_dataset.shape()[1], 0)


if __name__ == '__main__':
    unittest.main()