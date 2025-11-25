import unittest
import numpy as np

def tanimoto_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dot_product = np.dot(y, x)
    norm_x_sq = np.sum(x)
    norm_y_sq = np.sum(y, axis=1)
    denominator = norm_x_sq + norm_y_sq - dot_product
    denominator[denominator == 0] = 1.0
    similarity = dot_product / denominator
    return 1 - similarity


class TestTanimotoSimilarity(unittest.TestCase):

    def test_tanimoto_distance_calculation(self):
        
        x = np.array([1, 1, 1, 0, 0])
        y = np.array([
            [1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1],
            [1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0]
        ])

        expected_distances = np.array([
            0.0,
            1.0,
            0.5,
            1.0
        ])
        
        calculated_distances = tanimoto_similarity(x, y)

        self.assertTrue(np.allclose(expected_distances, calculated_distances))
        self.assertEqual(calculated_distances.shape, (4,))

    def test_tanimoto_zero_vector(self):
        
        x_zero = np.array([0, 0, 0])
        
        y_cases = np.array([
            [0, 0, 0],
            [1, 0, 0]
        ])
        
        expected_distances = np.array([1.0, 1.0])
        calculated_distances = tanimoto_similarity(x_zero, y_cases)

        self.assertTrue(np.allclose(expected_distances, calculated_distances))
        self.assertEqual(calculated_distances.shape, (2,))