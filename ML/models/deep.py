from torch import nn
from ML.types.trainable_model import TrainableModel

class DeepModel(TrainableModel):
    def __init__(self):
        super(DeepModel, self).__init__("deep")
        self.net = nn.Sequential(
            nn.Linear(2, 100), nn.LeakyReLU(),
            nn.Linear(100, 100), nn.LeakyReLU(),
            nn.Linear(100, 100), nn.LeakyReLU(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 3), nn.Tanh()
        )