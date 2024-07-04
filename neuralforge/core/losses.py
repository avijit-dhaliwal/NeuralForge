import numpy as np

class Loss:
    def __init__(self):
        pass

    def forward(self, y_true, y_pred):
        raise NotImplementedError

    def backward(self, y_true, y_pred):
        raise NotImplementedError

class MSE(Loss):
    def forward(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def backward(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size