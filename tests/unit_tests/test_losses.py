import os
from unittest import TestCase

import numpy as np
from si.io.data_file import read_data_file
from si.model_selection.split import train_test_split
from datasets import DATASETS_PATH
from si.neural_networks.losses import BinaryCrossEntropy, CategoricalCrossEntropy, MeanSquaredError


class TestLosses(TestCase):

    def setUp(self):
        
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')

        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_mean_squared_error_loss(self):

        error = MeanSquaredError().loss(self.dataset.y, self.dataset.y)

        self.assertEqual(error, 0)

    def test_mean_squared_error_derivative(self):

        derivative_error = MeanSquaredError().derivative(self.dataset.y, self.dataset.y)

        self.assertEqual(derivative_error.shape[0], self.dataset.shape()[0])

    def test_binary_cross_entropy_loss(self):
        error = BinaryCrossEntropy().loss(self.dataset.y, self.dataset.y)
        self.assertAlmostEqual(error, 0)
    
    def test_binary_cross_entropy_derivative(self): 
        derivative_error = BinaryCrossEntropy().derivative(self.dataset.y, self.dataset.y)
        self.assertEqual(derivative_error.shape[0], self.dataset.shape()[0])

    def test_categorical_cross_entropy_forward(self):
        loss = CategoricalCrossEntropy()
        
        y_true = np.array([[1, 0, 0], 
                           [0, 1, 0],
                           [0, 0, 1]])
        
        # Simulate predictions with some confidence
        y_pred = np.array([[0.9, 0.05, 0.05], 
                           [0.1, 0.8, 0.1],
                           [0.05, 0.05, 0.9]])
        
        result = loss.forward(y_true, y_pred)
        
        self.assertGreater(result, 0)
        self.assertAlmostEqual(result, 0.145, places=3)

    def test_categorical_cross_entropy_derivative(self):
        loss = CategoricalCrossEntropy()
        
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
        
        derivative = loss.derivative(y_true, y_pred)
        self.assertEqual(derivative.shape, y_pred.shape)
        self.assertLess(derivative[0, 0], 0)