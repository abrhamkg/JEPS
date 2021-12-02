import torch
import torch.nn as nn


class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(500, 400)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(400, 100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

if __name__ == '__main__':
    # Make data with batch size of 4
    X = torch.randn(4, 500)

    model = SimpleNeuralNetwork()

    out = model(X)

    print("Output shape", out.shape)