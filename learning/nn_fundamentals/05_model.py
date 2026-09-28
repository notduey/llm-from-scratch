import torch
import torch.nn as nn


# Define model
class SimpleNetwork(nn.Module): # nn.Module is PyTorch's base class for neural-network components

    # Constructor
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(1, 16) # 1 input -> 16 hidden values
        self.activation = nn.ReLU() # ReLU
        self.output = nn.Linear(16, 1) # 16 hidden values -> 1 prediction

    # Forward propagation
    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)

        return x

# Initialize model
model = SimpleNetwork()

# Input
x = torch.linspace(
    -2, # start
    2, # end
    100 # number of values
).reshape(-1, 1) # reshape to 2D tensor, 100 inputs each with 1 feature

# Target output
y = x ** 2 # y = x^2

# print model structure
print(model)

# Print model parameters
for name, parameter in model.named_parameters():
    print(name, parameter.shape)

# Loss function using Mean Squared Error (MSE)
loss_func = nn.MSELoss()

# Stochastic Gradient Descent as optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01 # learning rate
)

for epoch in range(1000):
    # Clear gradients from the previous training step to prevent calculating accumulated gradients
    optimizer.zero_grad()

    # Forward propagation
    prediction = model(x)

    # Calculate loss
    loss = loss_func(prediction, y)

    # Backpropagation
    loss.backward()

    # Update parameters using the calculated gradients
    optimizer.step()

    # Print progress occasionally (every 100 epochs)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}") # round to 4 decimals

print("\nFinal predictions:")
print(model(x))

print("\nTargets:")
print(y)

# Unseen inputs
test_x = torch.tensor([
    [-1.5],
    [-0.5],
    [0.5],
    [1.5]
])

test_prediction = model(test_x)

print("\nUnseen inputs:")
print(test_x)

print("\nPredictions on unseen inputs:")
print(test_prediction)