import torch
import torch.nn as nn

import matplotlib.pyplot as plt


class SimpleNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(1, 16)
        self.activation = nn.ReLU()
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)

        return x

# Full dataset
x = torch.linspace(-2, 2, 200).reshape(-1, 1)
y = x ** 2

# Training data
train_x = x[::2]
train_y = y[::2]

# Validation data
val_x = x[1::2]
val_y = y[1::2]

print("Training shape:", train_x.shape)
print("Validation shape:", val_x.shape)

# Initialize model
model = SimpleNetwork()

# Loss function
loss_func = nn.MSELoss()

# Optimizer (Stochastic GD)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

for epoch in range(1000):

    # Clear previous gradients
    optimizer.zero_grad()

    # Forward propagation on training data
    prediction = model(train_x)

    # Training loss
    train_loss = loss_func(prediction, train_y)

    # Backpropagation
    train_loss.backward()

    # Update parameters
    optimizer.step()

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch}: "
            f"Training Loss = {train_loss.item():.4f}"
        )

# Set model to evaluation mode, some layers behave different during training and eval
model.eval()

# Disable gradient calculation
with torch.no_grad():

    # Evaluate on validation data
    val_prediction = model(val_x)

    # Apply loss function between prediction and target
    val_loss = loss_func(
        val_prediction,
        val_y
    )

print("\nFinal training loss:", train_loss.item())
print("Validation loss:", val_loss.item())


# Generate points across the input range for plotting
plot_x = torch.linspace(-2, 2, 200).reshape(-1, 1)

model.eval()

with torch.no_grad():
    plot_prediction = model(plot_x)

# Convert tensors to NumPy arrays for Matplotlib
plot_x = plot_x.numpy()
plot_prediction = plot_prediction.numpy()

# Actual y = x^2 function
true_y = plot_x ** 2

plt.plot(plot_x, true_y, label="True: y = x²")
plt.plot(plot_x, plot_prediction, label="Model prediction")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Neural Network Approximation of y = x²")
plt.legend()

plt.show()