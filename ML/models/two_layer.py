from torch import nn
from ML.types.trainable_model import TrainableModel

class TwoLayerModel(TrainableModel):
    def __init__(self):
        super(TwoLayerModel, self).__init__("two_layer")
        self.net = nn.Sequential(
            nn.Linear(2, 5), nn.Tanh(),
            nn.Linear(5, 3), nn.Tanh()
        )
    
    def forward(self, input):
        return self.net(input)