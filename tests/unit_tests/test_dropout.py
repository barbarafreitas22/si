from unittest import TestCase
import numpy as np
from si.neural_networks.layers import Dropout

class TestDropout(TestCase):

    def test_dropout_layer_forward(self):
        """
        Tests the foward propagation in mode train and inference of the Dropout layer.
        """
        input_data = np.ones((10, 100)) 
        prob = 0.5
        dropout = Dropout(probability=prob)
        output_train = dropout.forward_propagation(input_data, training=True)
        
        # Verfify if some values are dropped
        self.assertTrue(np.any(output_train == 0))
        
        # Verfify if non-zero values are scaled correctly
        non_zeros = output_train[output_train != 0]
        self.assertTrue(np.allclose(non_zeros, 2.0))

        # Mode inference
        output_inference = dropout.forward_propagation(input_data, training=False)
        self.assertTrue(np.array_equal(output_inference, input_data))

    def test_dropout_layer_backward(self):
        """
        Tests the backward propagation of the Dropout layer.
        """
        input_data = np.array([[10, 20, 30]])
        dropout = Dropout(probability=0.5)
        
        # Forward 
        dropout.forward_propagation(input_data, training=True)
        
        # Simulate an output error
        output_error = np.ones_like(input_data)
        
        # Backward
        input_error = dropout.backward_propagation(output_error)
        self.assertTrue(np.all(input_error[dropout.mask == 0] == 0))