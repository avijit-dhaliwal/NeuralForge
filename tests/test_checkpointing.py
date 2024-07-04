# tests/test_checkpointing.py
import unittest
import os
import tempfile
from neuralforge.core.checkpointing import ModelCheckpoint
from neuralforge.core.model import Model

class TestCheckpointing(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.model = Model()  # Assume we have a simple model for testing

    def tearDown(self):
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_model_checkpoint(self):
        filepath = os.path.join(self.temp_dir, "model_epoch_{epoch:02d}.pkl")
        checkpointer = ModelCheckpoint(filepath, monitor='val_loss', mode='min', save_best_only=True)

        # Simulate some training epochs
        checkpointer(self.model, epoch=1, val_loss=0.5)
        self.assertTrue(os.path.exists(filepath.format(epoch=1)))

        checkpointer(self.model, epoch=2, val_loss=0.6)  # Worse loss, shouldn't save
        self.assertFalse(os.path.exists(filepath.format(epoch=2)))

        checkpointer(self.model, epoch=3, val_loss=0.4)  # Better loss, should save
        self.assertTrue(os.path.exists(filepath.format(epoch=3)))

if __name__ == '__main__':
    unittest.main()