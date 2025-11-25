import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())
    
    def test_dropna(self):
        """
        Tests the dropna method, checking for the removal of samples and the synchronization of the y vector.
        """
        # Test data: The rows 0 and 2 contain NaN
        X_with_nan = np.array([[1.0, 2.0, np.nan], 
                               [4.0, 5.0, 6.0], 
                               [7.0, np.nan, 9.0], 
                               [10.0, 11.0, 12.0]])
        y_original = np.array([10, 20, 30, 40])
        
        # Expected rows after dropna 
        X_expected = np.array([[4.0, 5.0, 6.0], 
                               [10.0, 11.0, 12.0]])
        y_expected = np.array([20, 40])
        
        dataset = Dataset(X_with_nan.copy(), y_original.copy())
        
        dataset.dropna()
        
        self.assertEqual((2, 3), dataset.shape())
        # Check if X and y contain the expected values
        self.assertTrue(np.array_equal(X_expected, dataset.X))
        self.assertTrue(np.array_equal(y_expected, dataset.y))

    def test_fillna_mean_median_and_constant(self):
        """
        Tests the fillna method with a value, 'mean', and 'median', and checks for invalid inputs.
        """
        X_with_nan = np.array([[1.0, 2.0, 3.0], 
                               [4.0, np.nan, 6.0], 
                               [7.0, 8.0, np.nan], 
                               [10.0, 12.0, 15.0]])
    

        #  Test with Constant Value (0.5) 
        dataset_const = Dataset(X_with_nan.copy())
        dataset_const.fillna(0.5)
        X_expected_const = np.array([[1.0, 2.0, 3.0], 
                                     [4.0, 0.5, 6.0], 
                                     [7.0, 8.0, 0.5], 
                                     [10.0, 12.0, 15.0]])
        self.assertTrue(np.array_equal(X_expected_const, dataset_const.X))
        
        # Test with mean
        dataset_mean = Dataset(X_with_nan.copy())
        dataset_mean.fillna("mean")
        X_expected_mean = np.array([[1.0, 2.0, 3.0], 
                                    [4.0, 7.33333333, 6.0], 
                                    [7.0, 8.0, 8.0], 
                                    [10.0, 12.0, 15.0]])
        # Use np.allclose for  float comparison
        self.assertTrue(np.allclose(X_expected_mean, dataset_mean.X))

        # Test with median
        dataset_median = Dataset(X_with_nan.copy())
        dataset_median.fillna("median")
        X_expected_median = np.array([[1.0, 2.0, 3.0], 
                                      [4.0, 8.0, 6.0], 
                                      [7.0, 8.0, 6.0], 
                                      [10.0, 12.0, 15.0]])
        self.assertTrue(np.array_equal(X_expected_median, dataset_median.X))

        # Test exceptions for invalid input
        dataset_invalid = Dataset(X_with_nan.copy())
        with self.assertRaises(ValueError):
            dataset_invalid.fillna("mode") # Test unsupported string
        with self.assertRaises(TypeError):
            dataset_invalid.fillna([1, 2]) # Test unsupported data type

    def test_remove_by_index(self):
        """
        Tests the remove_by_index method, checking the correct removal of a sample and exceptions.
        """
        X_original = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y_original = np.array([10, 20, 30])
        
        index_to_remove = 1 # Remove the middle row
        X_expected = np.array([[1, 2, 3], [7, 8, 9]])
        y_expected = np.array([10, 30])
        
        dataset = Dataset(X_original.copy(), y_original.copy())
        dataset.remove_by_index(index_to_remove)
        
        
        self.assertEqual((2, 3), dataset.shape())
        # Check if X and y contain the expected values
        self.assertTrue(np.array_equal(X_expected, dataset.X))
        self.assertTrue(np.array_equal(y_expected, dataset.y))

        # Test exception for indexs
        with self.assertRaises(IndexError):
            dataset.remove_by_index(100) # Index too high
        with self.assertRaises(IndexError):
            dataset.remove_by_index(-1)  # Invalid negative index