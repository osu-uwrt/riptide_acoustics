from ML.types.model import Model
import numpy as np

class RandomModel(Model):
    def evaluate(self, input) -> np.ndarray:
        return np.array([
            np.random.uniform(-1, 1),
            np.random.uniform(-1, 1),
            np.random.uniform(-1, 0)
        ])