# tests/test_schedulers.py
import unittest
from neuralforge.core.schedulers import StepLR, CosineAnnealingLR

class TestSchedulers(unittest.TestCase):
    def test_step_lr(self):
        scheduler = StepLR(initial_lr=0.1, step_size=3, gamma=0.1)
        self.assertAlmostEqual(scheduler.get_lr(0), 0.1)
        self.assertAlmostEqual(scheduler.get_lr(1), 0.1)
        self.assertAlmostEqual(scheduler.get_lr(3), 0.01)
        self.assertAlmostEqual(scheduler.get_lr(6), 0.001)

    def test_cosine_annealing_lr(self):
        scheduler = CosineAnnealingLR(initial_lr=0.1, T_max=10)
        self.assertAlmostEqual(scheduler.get_lr(0), 0.1)
        self.assertAlmostEqual(scheduler.get_lr(5), 0.05, places=2)
        self.assertAlmostEqual(scheduler.get_lr(10), 0.0)

if __name__ == '__main__':
    unittest.main()