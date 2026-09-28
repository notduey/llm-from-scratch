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
hidden = nn.Linear(
    in_features=1,
    out_features=4 # produce 4 outputs (arbitrary)
)

# 1 input, 4 internal values (and outputs)
output = hidden(x)

print("Input shape:", x.shape)
print("Weight shape:", hidden.weight.shape)
print("Bias shape:", hidden.bias.shape)
print("Output shape:", output.shape)

print(output)