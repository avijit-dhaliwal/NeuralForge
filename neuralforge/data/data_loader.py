import numpy as np
from PIL import Image

class DataLoader:
    def __init__(self, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def load_csv(self, filepath):
        data = np.genfromtxt(filepath, delimiter=',')
        if self.shuffle:
            np.random.shuffle(data)
        return data

    def load_images(self, directory, target_size=(224, 224)):
        images = []
        for filename in os.listdir(directory):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img = Image.open(os.path.join(directory, filename))
                img = img.resize(target_size)
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                images.append(img_array)
        return np.array(images)

    def batch_generator(self, data, labels):
        num_samples = len(data)
        indices = np.arange(num_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, num_samples, self.batch_size):
            end = min(start + self.batch_size, num_samples)
            batch_indices = indices[start:end]
            yield data[batch_indices], labels[batch_indices]