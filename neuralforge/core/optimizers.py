# neuralforge/core/optimizers.py

import numpy as np

class Optimizer:
    def __init__(self, learning_rate=0.01, clip_value=None):
        self.learning_rate = learning_rate
        self.clip_value = clip_value

    def clip_gradients(self, gradients):
        if self.clip_value is not None:
            for grad in gradients.values():
                np.clip(grad, -self.clip_value, self.clip_value, out=grad)
        return gradients

    def update(self, layer, gradients):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0, clip_value=None):
        super().__init__(learning_rate, clip_value)
        self.momentum = momentum
        self.velocity = {}

    def update(self, layer, gradients):
        gradients = self.clip_gradients(gradients)
        for param in ['weights', 'bias']:
            if param not in self.velocity:
                self.velocity[param] = np.zeros_like(getattr(layer, param))

            self.velocity[param] = self.momentum * self.velocity[param] - self.learning_rate * gradients[param]
            setattr(layer, param, getattr(layer, param) + self.velocity[param])

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, clip_value=None):
        super().__init__(learning_rate, clip_value)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layer, gradients):
        self.t += 1
        gradients = self.clip_gradients(gradients)
        for param in ['weights', 'bias']:
            if param not in self.m:
                self.m[param] = np.zeros_like(getattr(layer, param))
                self.v[param] = np.zeros_like(getattr(layer, param))

            grad = gradients[param]
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * np.square(grad)

            m_hat = self.m[param] / (1 - self.beta1**self.t)
            v_hat = self.v[param] / (1 - self.beta2**self.t)

            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            setattr(layer, param, getattr(layer, param) - update)

class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8, clip_value=None):
        super().__init__(learning_rate, clip_value)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = {}

    def update(self, layer, gradients):
        gradients = self.clip_gradients(gradients)
        for param in ['weights', 'bias']:
            if param not in self.cache:
                self.cache[param] = np.zeros_like(getattr(layer, param))

            grad = gradients[param]
            self.cache[param] = self.decay_rate * self.cache[param] + (1 - self.decay_rate) * np.square(grad)
            update = self.learning_rate * grad / (np.sqrt(self.cache[param]) + self.epsilon)
            setattr(layer, param, getattr(layer, param) - update)

class Adagrad(Optimizer):
    def __init__(self, learning_rate=0.01, epsilon=1e-8, clip_value=None):
        super().__init__(learning_rate, clip_value)
        self.epsilon = epsilon
        self.cache = {}

    def update(self, layer, gradients):
        gradients = self.clip_gradients(gradients)
        for param in ['weights', 'bias']:
            if param not in self.cache:
                self.cache[param] = np.zeros_like(getattr(layer, param))

            grad = gradients[param]
            self.cache[param] += np.square(grad)
            update = self.learning_rate * grad / (np.sqrt(self.cache[param]) + self.epsilon)
            setattr(layer, param, getattr(layer, param) - update)