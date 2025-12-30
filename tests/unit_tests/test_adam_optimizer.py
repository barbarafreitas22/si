from unittest import TestCase
import numpy as np
from si.neural_networks.optimizers import Adam

class TestOptimizers(TestCase):

    def test_adam_optimizer(self):
        w = np.array([10.0, 5.0]) # initial weights
        grad_loss_w = np.array([1.0, 1.0])
        learning_rate = 0.1
        
        optimizer = Adam(learning_rate=learning_rate)
        
        # Initial State 
        self.assertEqual(optimizer.t, 0)
        self.assertIsNone(optimizer.m)
        self.assertIsNone(optimizer.v)

        # First Update (t=1)
        new_w = optimizer.update(w, grad_loss_w)
        
        self.assertEqual(optimizer.t, 1)
        
        # m e v initialize
        self.assertIsNotNone(optimizer.m)
        self.assertIsNotNone(optimizer.v)
        
        self.assertFalse(np.array_equal(w, new_w))
        
        # If grad_loss_w is positive, new_w should be less than w
        self.assertLess(new_w[0], w[0])
        self.assertLess(new_w[1], w[1])
        
        # O shape must be preserved
        self.assertEqual(new_w.shape, w.shape)

        # Second Update (t=2)
        new_w_2 = optimizer.update(new_w, grad_loss_w)
        
        self.assertEqual(optimizer.t, 2)
        self.assertLess(new_w_2[0], new_w[0])

    def test_adam_shapes(self):
        """
        Tests if Adam optimizer correctly handles weight matrices of various shapes.
        """
        optimizer = Adam(learning_rate=0.01)
        
        # Weights matrix of shape (3, 2)
        w = np.zeros((3, 2))
        grad = np.ones((3, 2))
        
        new_w = optimizer.update(w, grad)
        
        self.assertEqual(new_w.shape, (3, 2))
        self.assertEqual(optimizer.m.shape, (3, 2))
        self.assertEqual(optimizer.v.shape, (3, 2))