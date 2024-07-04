# neuralforge/core/model.py

import numpy as np
from neuralforge.core.layers import Layer
from typing import Union
from neuralforge.core.optimizers import Optimizer, SGD, Adam, RMSprop, Adagrad
from neuralforge.core.losses import Loss
from neuralforge.core.callbacks import EarlyStopping
from neuralforge.core.schedulers import LRScheduler
from neuralforge.core.checkpointing import ModelCheckpoint

class Model:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        if isinstance(layer, Layer):
            self.layers.append(layer)
        else:
            raise ValueError("You can only add Layer objects to the model.")

    def compile(self, loss: Loss, optimizer: Union[Optimizer, SGD, Adam, RMSprop, Adagrad]):
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, X, training=True):
        output = X
        for layer in self.layers:
            if hasattr(layer, 'training'):
                output = layer.forward(output, training)
            else:
                output = layer.forward(output)
        return output

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            if hasattr(layer, 'weights'):
                self.optimizer.update(layer, {'weights': layer.weights_gradient, 'bias': layer.bias_gradient})

    def fit(self, X, y, epochs, batch_size=32, validation_data=None, 
            early_stopping=False, lr_scheduler=None, checkpointer=None):
        if early_stopping:
            early_stopper = EarlyStopping()

        n_samples = X.shape[1]
        n_batches = n_samples // batch_size

        for epoch in range(epochs):
            if lr_scheduler:
                current_lr = lr_scheduler.get_lr(epoch)
                self.optimizer.learning_rate = current_lr

            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            total_loss = 0
            for batch in range(n_batches):
                batch_indices = indices[batch*batch_size:(batch+1)*batch_size]
                X_batch = X[:, batch_indices]
                y_batch = y[:, batch_indices]

                output = self.forward(X_batch)
                loss = self.loss.forward(y_batch, output)
                total_loss += loss

                grad = self.loss.backward(y_batch, output)
                self.backward(grad)

            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}", end="")

            if validation_data:
                val_X, val_y = validation_data
                val_output = self.forward(val_X, training=False)
                val_loss = self.loss.forward(val_y, val_output)
                print(f", Val Loss: {val_loss:.4f}")

                if checkpointer:
                    checkpointer(self, epoch, val_loss=val_loss)

                if early_stopping and early_stopper(val_loss):
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            else:
                print()

    def predict(self, X):
        return self.forward(X, training=False)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        loss = self.loss.forward(y, predictions)
        return loss

    def save(self, filepath):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)