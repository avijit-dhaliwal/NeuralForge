# neuralforge/core/checkpointing.py
import os
import pickle

class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best = float('inf') if mode == 'min' else float('-inf')

    def __call__(self, model, epoch, **kwargs):
        current = kwargs.get(self.monitor)
        if current is None:
            return

        if (self.mode == 'min' and current < self.best) or (self.mode == 'max' and current > self.best):
            if self.save_best_only:
                self._save_model(model, epoch)
            self.best = current
        elif not self.save_best_only:
            self._save_model(model, epoch)

    def _save_model(self, model, epoch):
        filepath = self.filepath.format(epoch=epoch, **model.__dict__)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {filepath}")

