from unittest import TestCase
import numpy as np
import os
from si.io.csv_file import read_csv
from si.decomposition.pca import PCA
from si.data.dataset import Dataset 

class TestPCA(TestCase):
    
    def setUp(self):
        datasets_base_path = os.path.join('..', '..', 'datasets')
        self.csv_file = os.path.join(datasets_base_path, 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)
        self.n_components = 2
        self.pca = PCA(n_components=self.n_components)

    def test_fit(self):
        
        # Fit PCA to the dataset
        self.pca.fit(self.dataset)
        self.assertTrue(np.allclose(self.pca.mean, np.mean(self.dataset.X, axis=0)))
        # Expected shape is (n_components, n_features) due to .T in pca.py
        self.assertEqual(self.pca.components.shape, (self.n_components, self.dataset.X.shape[1]))

        # Ensure the number of explained variance entries matches the number of components
        self.assertEqual(len(self.pca.explained_variance), self.n_components)
        
        # New check: ensure explained variance is valid (between 0 and 1)
        self.assertTrue(np.all(self.pca.explained_variance >= 0))
        self.assertTrue(np.all(self.pca.explained_variance <= 1))

    def test_transform(self):
        self.pca.fit(self.dataset)
        dataset_reduced = self.pca.transform(self.dataset) 
        self.assertEqual(dataset_reduced.X.shape[0], self.dataset.X.shape[0])

        # Verify the number of components (features) in the reduced dataset
        self.assertEqual(dataset_reduced.X.shape[1], self.n_components)

        # Verify that the new feature names are correctly set 
        expected_features = [f"PC{i + 1}" for i in range(self.n_components)]
        self.assertListEqual(dataset_reduced.features, expected_features)


    def test_fit_invalid_components(self):
        """Tests if PCA raises a ValueError for invalid n_components."""
        # Test n_components < 1
        pca_invalid_low = PCA(n_components=0)
        with self.assertRaises(ValueError):
            pca_invalid_low.fit(self.dataset)
            
        pca_invalid_high = PCA(n_components=10) 
        with self.assertRaises(ValueError):
            pca_invalid_high.fit(self.dataset)
            
    def test_transform_unfitted(self):
        """Tests if transform raises a ValueError if not fitted."""
        pca_unfitted = PCA(n_components=2)
        with self.assertRaises(ValueError):
            pca_unfitted.transform(self.dataset)