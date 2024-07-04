import unittest
import numpy as np
from neuralforge.utils.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.data = np.random.randn(100, 10)
        self.labels = np.random.randint(0, 2, (100, 1))
        self.batch_size = 32
        self.data_loader = DataLoader(batch_size=self.batch_size)

    def test_batch_generator(self):
        generator = self.data_loader.batch_generator(self.data, self.labels)
        batch_x, batch_y = next(generator)
        self.assertEqual(batch_x.shape, (self.batch_size, 10))
        self.assertEqual(batch_y.shape, (self.batch_size, 1))

    def test_shuffle(self):
        generator1 = self.data_loader.batch_generator(self.data, self.labels)
        generator2 = self.data_loader.batch_generator(self.data, self.labels)
        batch1_x, _ = next(generator1)
        batch2_x, _ = next(generator2)
        self.assertFalse(np.array_equal(batch1_x, batch2_x))

if __name__ == '__main__':
    unittest.main()