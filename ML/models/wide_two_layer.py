from torch import nn
from ML.types.trainable_model import TrainableModel

class WideTwoLayerModel(TrainableModel):
    def __init__(self, num_inputs):
        super(WideTwoLayerModel, self).__init__("wide_two_layer")
        self.net = nn.Sequential(
            nn.Linear(num_inputs, 50), nn.Tanh(),
            nn.Linear(50, 3), nn.Tanh()
        )

    
    def forward(self, input):
        return self.net(input)