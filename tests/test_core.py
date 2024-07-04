# tests/test_core.py
import unittest
import numpy as np
from neuralforge.core.layers import Dense
from neuralforge.core.activations import relu, relu_prime
from neuralforge.core.model import Model

class TestCore(unittest.TestCase):
    def test_dense_layer(self):
        layer = Dense(2, 3)
        input_data = np.array([[1, 2]])
        output = layer.forward(input_data.T)
        self.assertEqual(output.shape, (3, 1))

    def test_relu_activation(self):
        x = np.array([-1, 0, 1])
        np.testing.assert_array_equal(relu(x), np.array([0, 0, 1]))

    def test_model_creation(self):
        model = Model()
        model.add(Dense(2, 3))
        model.add(relu)
        self.assertEqual(len(model.layers), 2)

if __name__ == '__main__':
    unittest.main()