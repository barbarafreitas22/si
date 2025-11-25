import unittest
import numpy as np
from si.neural_networks.layers import Dropout

class TestDropout(unittest.TestCase):
    """
    Unit tests for the Dropout layer, validating the Inverted Dropout mechanism.
    """

    def setUp(self):
        """Initial setup of the layer and test data."""
        np.random.seed(42)  
        self.input_data = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        self.dropout_rate = 0.5
        self.dropout = Dropout(probability=self.dropout_rate)

    def test_structural_properties(self):
        """Verifies parameter count."""
        self.assertEqual(0, self.dropout.parameters())

    def test_forward_propagation_inference_mode(self):
        """Verifies that input is returned UNCHANGED during inference (training=False)."""
        output = self.dropout.forward_propagation(self.input_data, training=False)
        
        # Output needs to be numerically identical to the input
        self.assertTrue(np.array_equal(self.input_data, output))

    def test_forward_propagation_training_mode_scaling(self):
        """
        Verifies that during training:
        1. Zeros are applied via mask.
        2. The active neurons are scaled by 1/(1-p).
        """
        output = self.dropout.forward_propagation(self.input_data, training=True)
        
        self.assertIsNotNone(self.dropout.mask)
        
        # 2. Check Scaling Math 
        scaling_factor = 1.0 / (1.0 - self.dropout_rate)
        active_mask = output != 0
        expected_values = self.input_data[active_mask] * scaling_factor
        active_outputs = output[active_mask]
        
        self.assertTrue(np.allclose(active_outputs, expected_values),
                        "Os valores ativos não foram escalados corretamente segundo o Inverted Dropout.")

    def test_backward_propagation_applies_mask(self):
        """Verifies that the gradient is multiplied by the stored mask."""
        # Run forward pass to generate the mask 
        self.dropout.forward_propagation(self.input_data, training=True)
        
        # Define a simple output error gradient
        output_error = np.ones_like(self.input_data)
        
        #  Calculate the backward pass
        input_error = self.dropout.backward_propagation(output_error)
        
        # Verify logic: Error should be 0 where output was 0
        dropped_indices = (self.dropout.output == 0)
        self.assertTrue(np.all(input_error[dropped_indices] == 0))
        
        # Verify active indices match the mask
        active_indices = (self.dropout.output != 0)
        self.assertTrue(np.all(input_error[active_indices] == output_error[active_indices]))

    def test_output_shape(self):
        """Verifies output shape (must run forward first)."""
        self.dropout.forward_propagation(self.input_data, training=True)
        self.assertEqual(self.input_data.shape, self.dropout.output_shape())

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)