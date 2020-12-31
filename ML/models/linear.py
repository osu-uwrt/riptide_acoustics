from torch import nn
from ML.types.trainable_model import TrainableModel

class LinearModel(TrainableModel):
    def __init__(self, num_inputs):
        super(LinearModel, self).__init__("linear")
        self.net = nn.Sequential(
            nn.Linear(num_inputs, 3)
        )
    
    def forward(self, input):
        return self.net(input)