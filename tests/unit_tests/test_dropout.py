import unittest
import numpy as np
from typing import Tuple, Optional, Any

class TestDropout(unittest.TestCase):
    """
    Unit tests for the Dropout layer, validating the Inverted Dropout mechanism.
    """

    def setUp(self):
        """Initial setup of the layer and test data."""
        self.input_data = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        self.input_shape = self.input_data.shape
        self.dropout_rate = 0.5
        self.dropout = Dropout(probability=self.dropout_rate)

    def test_structural_properties(self):
        """Verifies output shape and parameter count."""
        self.assertEqual(self.input_shape, self.dropout.output_shape(self.input_shape))
        self.assertEqual(0, self.dropout.parameters())

    def test_forward_propagation_inference_mode(self):
        """Verifies that input is returned UNCHANGED during inference (training=False)."""
        output = self.dropout.forward_propagation(self.input_data, training=False)
        
        # Output must be numerically identical to input
        self.assertTrue(np.array_equal(self.input_data, output))
        self.assertIsNone(self.dropout.mask, "Mask should not be generated in inference mode.")

    def test_forward_propagation_training_mode_scaling(self):
        """
        Verifies that during training:
        1. Zeros are applied via mask.
        2. The expected sum is preserved (property of Inverted Dropout).
        """
        output = self.dropout.forward_propagation(self.input_data, training=True)
        
        # 1. Assert mask was created and applied
        self.assertIsNotNone(self.dropout.mask)
        self.assertTrue(np.any(output == 0.0), "Dropout failed: output must contain zeros.")

        # 2. Check Sum Preservation (Inverted Dropout property)
        # Expected Sum = Sum of original input data
        expected_sum = np.sum(self.input_data)
        actual_sum = np.sum(output)
        
        # Due to the random nature of binomial sampling, we use a tolerance (15%)
        self.assertAlmostEqual(expected_sum, actual_sum, delta=expected_sum * 0.15,
                               msg="Output sum is not preserved/scaled correctly, indicating an error in Inverted Dropout.")

    def test_backward_propagation_applies_mask(self):
        """Verifies that the gradient is multiplied by the stored mask."""
        # 1. Run forward pass to generate the mask (training=True)
        self.dropout.forward_propagation(self.input_data, training=True)
        
        # 2. Define a simple output error gradient
        output_error = np.ones_like(self.input_data) * 0.5
        
        # 3. Calculate the backward pass
        input_error = self.dropout.backward_propagation(output_error)
        
        # Verify input_error is equal to output_error * mask (which includes scaling)
        expected_error = output_error * self.dropout.mask * (1.0 / (1.0 - self.dropout_rate)) 
        
        # NOTE: The backward implementation in the provided code is `output_error * self.mask`.
        # This works because the mask already has the scaling embedded from the forward pass.
        expected_error_simplified = output_error * self.dropout.mask
        
        self.assertTrue(np.array_equal(input_error, expected_error_simplified), 
                        "Backward pass failed: gradient was not correctly masked.")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)