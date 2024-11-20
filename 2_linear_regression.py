import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import optimizer

# Generate synthetic data
np.random.seed(1337)
true_bias = 4
true_slope = 3
x = 2 * np.random.rand(100, 1)
y = true_bias + true_slope * x + np.random.randn(100, 1) * 0.5  # Reduced noise by multiplying with 0.5

# Convert data to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Define the neural network
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # Define the linear layer with one input and one output, fully connected.
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = LinearRegressionModel()

# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
number_epochs = 1000
for epoch in range(number_epochs):
    model.train()

    # Forward pass
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward pass and optimisation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{number_epochs}], Loss: {loss.item():.4f}')

# Check if the optimizer is correctly defined
if not isinstance(optimizer, optim.Optimizer):
    raise TypeError("The optimizer is not correctly defined. Please check the optimizer definition.")

# Make predictions
model.eval()
with torch.no_grad():
    y_pred = model(x_tensor).numpy()

# Visualize the data and results
plt.figure(figsize=(10, 5))

# Plot the synthetic data
plt.subplot(1, 2, 1)
plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Synthetic Data with Reduced Noise')

# Plot the linear regression results
plt.subplot(1, 2, 2)
plt.scatter(x, y, label='Data')
plt.plot(x, y_pred, color='red', label='Linear Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression using Neural Network')
plt.legend()

# Show all plots at once
plt.tight_layout()
plt.show()

# Print the true parameters
print(f'True slope: {true_slope}')
print(f'True bias: {true_bias}')

# Print the learned parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f'{name}: {param.data.item()}')

# Compare the true and learned parameters
learned_slope = model.linear.weight.item()
learned_bias = model.linear.bias.item()
print(f'Learned slope: {learned_slope}')
print(f'Learned bias: {learned_bias}')
