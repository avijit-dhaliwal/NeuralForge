import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.bias = np.zeros((output_size, 1))

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class BatchNorm(Layer):
    def __init__(self, input_shape, epsilon=1e-8, momentum=0.9):
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = np.ones(input_shape)
        self.beta = np.zeros(input_shape)
        self.moving_mean = np.zeros(input_shape)
        self.moving_var = np.ones(input_shape)

    def forward(self, input, training=True):
        if training:
            mean = np.mean(input, axis=0)
            var = np.var(input, axis=0)
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * mean
            self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * var
        else:
            mean = self.moving_mean
            var = self.moving_var

        self.input = input
        self.normalized = (input - mean) / np.sqrt(var + self.epsilon)
        return self.gamma * self.normalized + self.beta

    def backward(self, output_gradient):
        input_mean = np.mean(self.input, axis=0)
        input_var = np.var(self.input, axis=0)
        m = output_gradient.shape[0]

        normalized_gradient = output_gradient * self.gamma
        var_gradient = np.sum(normalized_gradient * (self.input - input_mean), axis=0) * -0.5 * (input_var + self.epsilon)**(-1.5)
        mean_gradient = np.sum(normalized_gradient * -1 / np.sqrt(input_var + self.epsilon), axis=0) + var_gradient * np.mean(-2 * (self.input - input_mean), axis=0)

        self.gamma_gradient = np.sum(output_gradient * self.normalized, axis=0)
        self.beta_gradient = np.sum(output_gradient, axis=0)
        input_gradient = normalized_gradient / np.sqrt(input_var + self.epsilon) + var_gradient * 2 * (self.input - input_mean) / m + mean_gradient / m

        return input_gradient