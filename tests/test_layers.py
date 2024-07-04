import unittest
import numpy as np
from neuralforge.core.layers import Dense, Convolutional

class TestLayers(unittest.TestCase):
    def test_dense_layer(self):
        layer = Dense(5, 3)
        input_data = np.random.randn(5, 1)
        output = layer.forward(input_data)
        self.assertEqual(output.shape, (3, 1))

    def test_convolutional_layer(self):
        layer = Convolutional((3, 32, 32), kernel_size=3, depth=16)
        input_data = np.random.randn(3, 32, 32)
        output = layer.forward(input_data)
        self.assertEqual(output.shape, (16, 30, 30))

if __name__ == '__main__':
    unittest.main()