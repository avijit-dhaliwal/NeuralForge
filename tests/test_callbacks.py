# tests/test_callbacks.py
import unittest
from neuralforge.core.callbacks import EarlyStopping

class TestCallbacks(unittest.TestCase):
    def test_early_stopping(self):
        early_stopper = EarlyStopping(patience=3, min_delta=0.1)
        
        # Simulating decreasing loss
        self.assertFalse(early_stopper(10.0))
        self.assertFalse(early_stopper(9.0))
        self.assertFalse(early_stopper(8.0))
        
        # Simulating stagnating loss
        self.assertFalse(early_stopper(7.95))
        self.assertFalse(early_stopper(7.90))
        self.assertTrue(early_stopper(7.85))  # Should trigger early stopping

if __name__ == '__main__':
    unittest.main()