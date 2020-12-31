import sys
sys.path.append(".")

from ML.models import *
from ML.generator import Generator
from ML.ping_generator import PingGenerator
import numpy as np
import matplotlib.pyplot as plt
import torch
from ML.types.trainable_model import TrainableModel
from ML.utils import calculate_error

generator = PingGenerator()

##################################################################
# This file is used to test a model against a set of samples
# Enter the name of the model here
model = DeepModel(generator.get_num_inputs())
##################################################################

# Prepare our model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if isinstance(model, TrainableModel):
    model.load()
    model = model.to(device)

truth_list = []
predicted_list = []
error_list = []
SAMPLES = 1000 # Number of samples to run
inputs, truths = generator.generate_samples(size=SAMPLES)

for sample_index in range(SAMPLES):
    # Generate sample
    input, truth = inputs[sample_index], truths[sample_index]

    # Run it through the model
    if isinstance(model, TrainableModel):
        predicted = model.evaluate(input, device=device)
    else:
        predicted = model.evaluate(input)
        
    # Record the error
    truth_list.append(truth)
    predicted_list.append(predicted)
    error_list.append(calculate_error(truth, predicted))

# Show results
print("Median error: %0.2f degrees" % np.median(error_list))
ax = plt.boxplot(error_list)
plt.ylabel("Error (degrees)")
plt.show()

