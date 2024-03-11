import sys

# Add the directory containing your module to the Python path
sys.path.append("./")

import torch
import torch.nn as nn
import torchTensorRef

# from torchTensorRef import torch
# nn = torch.nn

device = "cpu"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    print("MPS device not found.")

torchTensorRef.tensorsManager.device = device

# Create two tensors of size 2x3 filled with random values
tensor_a = torch.rand(2, 3)
tensor_b = torch.rand(2, 3)

print("Tensor A:\n", tensor_a.target)
print("Tensor B:\n", tensor_b.target)

# Sum the two tensors
tensor_sum = tensor_a + tensor_b

print("Sum of Tensor A and B:\n", tensor_sum.target)

### RNN testing


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.linear1 = nn.Linear(
            in_features=10, out_features=20
        )  # Example input features: 10, output features: 20
        self.prelu = nn.PReLU()  # PReLU activation
        self.linear2 = nn.Linear(
            in_features=20, out_features=1
        )  # Output layer for binary classification

    def forward(self, x):
        x = self.linear1(x)
        x = self.prelu(x)  # Applying PReLU activation
        x = self.linear2(x)
        return torch.sigmoid(x)  # Applying sigmoid activation for binary classification


# Create a model instance
model = SimpleNN()

# Example input tensor
input_tensor = torch.randn(5, 10)  # Batch size of 5, feature size of 10

# Forward pass through the model
output = model(input_tensor)

print(output.target)  # Print the output of the model

print("End of test")
