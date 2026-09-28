import torch
import torch.nn as nn


# tensor
x = torch.tensor([
    [-2.0],
    [-1.0],
    [0.0],
    [1.0],
    [2.0]
])

# linear layer, pytorch auto-initializes weights and biases randomly
layer = nn.Linear(in_features=1, out_features=1) # y = Wx + b

# get output
output = layer(x) # multiply input (x) by weights and add bias

print(output)
print(output.shape)

# trainable parameter tensors
print("Weight:", layer.weight)
print("Bias:", layer.bias)

print("Weight shape:", layer.weight.shape)
print("Bias shape:", layer.bias.shape)