import unittest
import numpy as np
from neuralforge.core.losses import MSE, CrossEntropy

class TestLosses(unittest.TestCase):
    def test_mse(self):
        mse = MSE()
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 2.1, 2.9])
        loss = mse.forward(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.0100, places=4)

    def test_cross_entropy(self):
        ce = CrossEntropy()
        y_true = np.array([0, 1, 0])
        y_pred = np.array([0.1, 0.8, 0.1])
        loss = ce.forward(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.2231, places=4)

if __name__ == '__main__':
    unittest.main()