import torch


# Create 2D tensors
x = torch.tensor([ # input
    [-2.0],
    [-1.0],
    [0.0],
    [1.0],
    [2.0]
])

y = torch.tensor([ # target output
    [4.0],
    [1.0],
    [0.0],
    [1.0],
    [4.0]
])

print(x) # entire tensor
print(y)

print(f"x shape: {x.shape}") # [training examples, features per example]
print(f"y shape: {y.shape}")
print(x.dtype)

print(f"First training example: x = {x[0]}, y = {y[0]}")
print(x[0][0]) # individual tensor value
print(x[0][0].item()) # value as python float