import numpy as np
import math
import torch

# Object that will generate sample data for the conversion of azimuth 
# and altitude to a unit vector pointing in the correct direction.
# Mathematically that is
# [az, al] -> [cos(az)cos(al), sin(az)cos(al), sin(al)]
# For all az and al in the ranges [0, 2π] and [-π/2, 0] respectively
# Where an altitude of 0 is on the xy plane and an altitude of -π/2 
# equals a z of -1

class Generator:
    def generate_sample(self):
        azimuth = np.random.uniform(0, 2 * math.pi)
        altitude = np.random.uniform(-math.pi/2, 0)

        x = math.cos(azimuth)*math.cos(altitude)
        y = math.sin(azimuth)*math.cos(altitude)
        z = math.sin(altitude)

        return np.array([azimuth, altitude]), np.array([x, y, z])

    def generate_samples(self, size=64):
        inputs = []
        truths = []
        for _ in range(size):
            input, truth = self.generate_sample()
            inputs.append(input)
            truths.append(truth)

        return inputs, truths

    def generate_tensor_samples(self, device=None, size=64):
        if device is None:
            device = torch.device("cpu")

        # Generate samples
        inputs, truths = self.generate_samples(size)

        # Output as Tensors
        return torch.Tensor(inputs).to(device), torch.Tensor(truths).to(device)

        
