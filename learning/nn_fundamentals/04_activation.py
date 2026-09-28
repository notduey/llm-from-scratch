import torch
import torch.nn as nn


x = torch.tensor([
    [-2.0],
    [-1.0],
    [0.0],
    [1.0],
    [2.0]
])

# Hidden layer
hidden = nn.Linear(1, 4)

# Activation function
activation = nn.ReLU() # we use ReLU

# Output layer
output_layer = nn.Linear(4, 1) # 4 hidden features -> 1 output

# 1 input feature -> 4 internal hidden features
hidden_output = hidden(x)

# Apply activation
activated_output = activation(hidden_output)

# turn internal features -> 1 prediction
prediction = output_layer(activated_output)

print("Hidden output:")
print(hidden_output)

print("\nAfter ReLU:")
print(activated_output)

print("\nPrediction:")
print(prediction)