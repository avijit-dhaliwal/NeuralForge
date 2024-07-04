import unittest
import numpy as np
from neuralforge.core.model import Model
from neuralforge.core.layers import Dense
from neuralforge.core.activations import relu
from neuralforge.core.losses import MSE

class TestModel(unittest.TestCase):
    def test_model_creation_and_prediction(self):
        model = Model()
        model.add(Dense(2, 3))
        model.add(relu)
        model.add(Dense(3, 1))
        model.set_loss(MSE())

        input_data = np.array([[1, 2]])
        output = model.predict(input_data.T)
        self.assertEqual(output.shape, (1, 1))

    def test_model_training(self):
        model = Model()
        model.add(Dense(2, 3))
        model.add(relu)
        model.add(Dense(3, 1))
        model.set_loss(MSE())

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
        y = np.array([[0], [1], [1], [0]])

        initial_loss = model.loss.forward(y, model.predict(X))
        model.fit(X, y, epochs=100, learning_rate=0.1)
        final_loss = model.loss.forward(y, model.predict(X))

        self.assertLess(final_loss, initial_loss)

if __name__ == '__main__':
    unittest.main()