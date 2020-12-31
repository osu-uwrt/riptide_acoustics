from ML.types.model import Model
import numpy as np

class NoSkillModel(Model):
    def evaluate(self, input) -> np.ndarray:
        return np.array([0, 0, -.5])