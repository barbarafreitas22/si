from unittest import TestCase
import numpy as np


from si.neural_networks.activation import TanhActivation, SoftmaxActivation

class TestActivation(TestCase):

    def test_tanh_activation(self):
        """
        Testa a função Tanh e a sua derivada.
        """
        layer = TanhActivation()
        
        # Test Forward (Activation Function) 
        input_data = np.array([[-10.0, -1.0, 0.0, 1.0, 10.0]])
        
        result = layer.activation_function(input_data)
        
        # Tanh smashes values beetween -1 and 1
        self.assertTrue(np.all(result >= -1))
        self.assertTrue(np.all(result <= 1))
        
        # Tanh(0) must be 0
        self.assertEqual(result[0, 2], 0)
        
        # Compare with numpy's tanh
        expected_result = np.tanh(input_data)
        self.assertTrue(np.allclose(result, expected_result))

        # Test Backward (Derivative) 
        # define self.output manually (ActivationLayer would do in forward_propagation)
        layer.output = result 
        
        derivative = layer.derivative(input_data)
        
        # Formula: 1 - tanh^2(x)
        expected_derivative = 1 - expected_result ** 2
        
        self.assertTrue(np.allclose(derivative, expected_derivative))

    def test_softmax_activation(self):
        """
        Testa a função Softmax (soma=1) e a estabilidade numérica.
        """
        layer = SoftmaxActivation()
        
        #  Test Forward (Activation Function) 
        input_data = np.array([[1.0, 2.0, 3.0], 
                               [2.0, 2.0, 2.0]])
        
        result = layer.activation_function(input_data)
        
        # Sum of probabilities for each sample should be 1
        row_sums = np.sum(result, axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0))
        
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))

        # Test numeric stability, if we use very large values would cause overflow (infinite)
        huge_input = np.array([[1000.0, 1001.0, 1002.0]])
        
        try:
            stable_result = layer.activation_function(huge_input)
            # Ensure the output is still a valid probability distribution
            self.assertTrue(np.allclose(np.sum(stable_result, axis=1), 1.0))
            print("\nSoftmax Stability Test: PASSED (No Overflow)")
        except Exception as e:
            self.fail(f"Softmax falhou com números grandes: {e}")

        # Test Backward (Derivative) 
        layer.output = result
        
        derivative = layer.derivative(input_data)
        
        # Simplified derivative: f'(x) = f(x) * (1 - f(x))
        expected_derivative = result * (1 - result)
        
        self.assertTrue(np.allclose(derivative, expected_derivative))