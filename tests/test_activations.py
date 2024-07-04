import unittest
import numpy as np
from neuralforge.core.activations import relu, sigmoid, tanh

class TestActivations(unittest.TestCase):
    def test_relu(self):
        x = np.array([-1, 0, 1])
        np.testing.assert_array_equal(relu(x), np.array([0, 0, 1]))

    def test_sigmoid(self):
        x = np.array([0])
        self.assertAlmostEqual(sigmoid(x)[0], 0.5, places=7)

    def test_tanh(self):
        x = np.array([0])
        self.assertAlmostEqual(tanh(x)[0], 0, places=7)

if __name__ == '__main__':
    unittest.main()