from torch import nn
import torch
import numpy as np

class Model:
    def __init__(self):
        pass

    def evaluate(self, input) -> np.ndarray:
        raise NotImplementedError

class TrainableModel(torch.nn.Module, Model):
    def __init__(self, name):
        super(TrainableModel, self).__init__()
        self.name = name

    def file_name(self):
        return self.name + ".pt"

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


class DeepModel(TrainableModel):
    def __init__(self, num_inputs):
        super(DeepModel, self).__init__("deep")
        self.net = nn.Sequential(
            nn.Linear(num_inputs, 100), nn.LeakyReLU(),
            nn.Linear(100, 100), nn.LeakyReLU(),
            nn.Linear(100, 100), nn.LeakyReLU(),
            nn.Linear(100, 100), nn.Tanh(),
            nn.Linear(100, 3), nn.Tanh()
        )