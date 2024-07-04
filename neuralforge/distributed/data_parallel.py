import ray

@ray.remote
class ParameterServer:
    def __init__(self, model):
        self.model = model

    def get_weights(self):
        return [layer.weights for layer in self.model.layers if hasattr(layer, 'weights')]

    def update_weights(self, gradients):
        for layer, grad in zip([layer for layer in self.model.layers if hasattr(layer, 'weights')], gradients):
            layer.weights -= grad

@ray.remote
class Worker:
    def __init__(self, model):
        self.model = model

    def compute_gradients(self, data_batch):
        gradients = []
        for x, y in data_batch:
            output = self.model.predict(x)
            grad = self.model.loss.backward(y, output)
            for layer in reversed(self.model.layers):
                if hasattr(layer, 'weights'):
                    weights_gradient = np.dot(grad, layer.input.T)
                    gradients.append(weights_gradient)
                grad = layer.backward(grad, 0)  # learning_rate = 0 as we're not updating weights here
        return gradients

def distribute_training(model, data, num_workers, epochs, learning_rate):
    ray.init()
    ps = ParameterServer.remote(model)
    workers = [Worker.remote(model) for _ in range(num_workers)]

    for epoch in range(epochs):
        futures = [w.compute_gradients.remote(data_batch) for w, data_batch in zip(workers, np.array_split(data, num_workers))]
        gradients = ray.get(futures)
        average_gradients = [np.mean(grad, axis=0) for grad in zip(*gradients)]
        ps.update_weights.remote(average_gradients)
        
        weights = ray.get(ps.get_weights.remote())
        for worker in workers:
            worker.set_weights.remote(weights)

    ray.shutdown()

