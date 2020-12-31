import numpy as np
import math
import torch
import search_test

# Object that will generate sample data for the conversion of azimuth 
# and altitude to a unit vector pointing in the correct direction.
# Mathematically that is
# [az, al] -> [cos(az)cos(al), sin(az)cos(al), sin(al)]
# For all az and al in the ranges [0, 2π] and [-π/2, 0] respectively
# Where an altitude of 0 is on the xy plane and an altitude of -π/2 
# equals a z of -1


class PingGenerator:
    def get_num_inputs(self):
        return 4

    def generate_samples(self, size=64) -> tuple:
        inputs = []
        truths = []
        sample_data = search_test.generate_samples(size)
        for sample in sample_data:
            inputs.append(np.array([sample["mic_spacing"] * 100, sample["ping_frequency"]/10000, sample["x_phase_difference"], sample["y_phase_difference"]]))
            truths.append(sample["ping_direction"])

        return inputs, truths

    def generate_tensor_samples(self, device=None, size=64):
        if device is None:
            device = torch.device("cpu")

        # Generate samples
        inputs, truths = self.generate_samples(size)

        # Output as Tensors
        return torch.Tensor(inputs).to(device), torch.Tensor(truths).to(device)

        
