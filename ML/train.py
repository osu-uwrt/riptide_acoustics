import sys
sys.path.append(".")

from ML.models import *
from ML.generator import Generator
from ML.utils import calculate_error
import numpy as np
import matplotlib.pyplot as plt
import torch

##################################################################
# This file is used to train a trainable model
# Enter the name of the model here
model = DeepModel()
##################################################################

generator = Generator()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup plots
plt.ion()
fig, ax_loss = plt.subplots()
ax_error = ax_loss.twinx()
ax_loss.set_xlabel("Batch")
ax_loss.set_ylabel("Loss")
error_plot_color = 'tab:red'
ax_error.set_ylabel("Error (degrees)", color=error_plot_color)
ax_error.tick_params(axis='y', labelcolor=error_plot_color)
plt.show()

# Initialize net, optimizer, and loss
net = model.to(device)
optimizer = torch.optim.Adam(net.parameters())
criterion = torch.nn.MSELoss()

# Initialize our first batch
inputs, truths = generator.generate_tensor_samples(device)
iteration = 0
best_error = 10000
last_error_update = 0
losses = []
errors = []

# While we have not had a long period without getting better
while iteration - last_error_update < 2000:
    # Get the data
    net.train()
    
    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, truths)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

    # Generate a new batch and predict
    inputs, truths = generator.generate_tensor_samples(device)
    labels_numpy = truths.cpu().detach()
    guess = net(inputs).cpu()
    guess_numpy = guess.detach()
    
    # Get average error
    batch_errors = []
    for i in range(len(guess_numpy)):
        batch_errors.append(calculate_error(labels_numpy[i], guess_numpy[i]))
    error = np.average(batch_errors)
    errors.append(error)
    

    # If new lowest error, save net
    if error < best_error:
        best_error = error
        last_error_update = iteration
        net.save()
        print("New Best! Model saved.")
        print("Average Batch Error: %0.2f degrees" % error)
    
    # Plot batch and loss
    if iteration % 100 == 0:
        ax_loss.clear()
        ax_error.clear()
        ax_loss.set_xlabel("Batch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_ylim(0, 0.5)
        ax_error.set_ylim(0, 70)
        ax_error.set_ylabel("Error (degrees)", color=error_plot_color)
        ax_loss.plot(range(iteration+1), losses)
        ax_error.plot(range(iteration+1), errors, color=error_plot_color)
        plt.draw()
        plt.pause(0.001)

    iteration += 1

print("Best error: %f" % best_error)