import unittest
import numpy as np
from neuralforge.core.optimizers import SGD, Adam
from neuralforge.core.layers import Dense

class TestOptimizers(unittest.TestCase):
    def test_sgd(self):
        layer = Dense(2, 1)
        layer.weights = np.array([[1.0], [2.0]])
        layer.bias = np.array([[0.5]])
        optimizer = SGD(learning_rate=0.1)
        gradients = {'weights': np.array([[0.1], [0.2]]), 'bias': np.array([[0.05]])}
        optimizer.update(layer, gradients)
        np.testing.assert_array_almost_equal(layer.weights, np.array([[0.99], [1.98]]))
        np.testing.assert_array_almost_equal(layer.bias, np.array([[0.495]]))

    def test_adam(self):
        layer = Dense(2, 1)
        layer.weights = np.array([[1.0], [2.0]])
        layer.bias = np.array([[0.5]])
        optimizer = Adam(learning_rate=0.001)
        gradients = {'weights': np.array([[0.1], [0.2]]), 'bias': np.array([[0.05]])}
        optimizer.update(layer, gradients)
        # Note: The exact values will depend on the Adam implementation details
        self.assertNotEqual(layer.weights[0, 0], 1.0)
        self.assertNotEqual(layer.weights[1, 0], 2.0)
        self.assertNotEqual(layer.bias[0, 0], 0.5)

if __name__ == '__main__':
    unittest.main()