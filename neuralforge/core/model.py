class Model:
    def __init__(self):
        self.layers = []
        self.loss = None

    def add(self, layer):
        self.layers.append(layer)

    def set_loss(self, loss):
        self.loss = loss

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def fit(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                output = self.predict(x)
                error += self.loss.forward(y, output)
                grad = self.loss.backward(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)
            error /= len(x_train)
            print(f'Epoch {epoch+1}/{epochs}, Error: {error}')