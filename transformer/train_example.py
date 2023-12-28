import torch
import torch.nn as nn
import torch.optim as optim

# Data Preparation
# Create some synthetic data for input and output
# Assuming we have 100 samples, each sample is 10-dimensional
X = torch.randn(100, 10)
Y = torch.randn(100, 1)

# Define a Simple Fully Connected Network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # First fully connected layer
        self.fc2 = nn.Linear(5, 1)   # Second fully connected layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Use ReLU activation function
        x = self.fc2(x)
        return x

# Instantiate the network
net = SimpleNet()

# Define the loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error as the loss function
optimizer = optim.SGD(net.parameters(), lr=0.01)  # Stochastic Gradient Descent as the optimizer

# Train the network
epochs = 100
for epoch in range(epochs):
    # Forward pass
    outputs = net(X)
    loss = criterion(outputs, Y)

    # Backward pass and optimization
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Backpropagation
    optimizer.step()       # Update parameters

    # Print statistics
    if (epoch + 1) % 10 == 0:  # Print every 10 epochs
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

print('Training completed')