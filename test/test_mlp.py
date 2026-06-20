import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = torch.nn.Sequential(

            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits
    

model = NeuralNetwork(num_inputs=50, num_outputs=3)
print(model)

# Count the number of trainable parameters in the model
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)

print(model.layers[0].weight)
print(model.layers[0].weight.shape)

# Set the random seed for reproducibility
torch.manual_seed(123)
model = NeuralNetwork(num_inputs=50, num_outputs=3)
print(model.layers[0].weight)

# toy example of a forward pass through the model
torch.manual_seed(123)
X = torch.rand((1, 50))
out = model(X)
print(out)

# Disable gradient tracking for inference
with torch.no_grad():
    out = model(X)
print(out)

# Apply softmax to the output logits to get probabilities
with torch.no_grad():
    out = torch.softmax(model(X), dim=1)
print(out)