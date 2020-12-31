from numpy.lib.npyio import load
import torch
import numpy as np
from ML.types.model import Model

class TrainableModel(torch.nn.Module, Model):
    def __init__(self, name):
        super(TrainableModel, self).__init__()
        self.name = name

    def file_name(self):
        return "ML/models/" + self.name + ".pt"

    def forward(self, input):
        return self.net(input)

    def evaluate(self, inputs, device=None) -> np.ndarray:
        if device is None:
            device = torch.device("cpu")

        outputs = self(torch.Tensor(inputs).to(device))
        return outputs.cpu().detach().numpy()

    def save(self):
        torch.save(self.state_dict(), self.file_name())

    def load(self):
        self.load_state_dict(torch.load(self.file_name()))
        self.eval()