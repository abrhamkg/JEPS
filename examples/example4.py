import torch
import torch.nn as nn
import examples.example3
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class RandomDataDataset(Dataset):
    def __init__(self):
        self.X = torch.randn((5000, 500))
        self.y = torch.randint(0, 1, (5000,))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def train_model(dataloader):
    num_epochs = 5
    model = examples.example3.SimpleNeuralNetwork()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        for i, batch in enumerate(dataloader):
            x, y = batch
            out = model(x)

            loss = criterion(out, y)

            loss.backward()

            optimizer.step()
            if i % 10 == 0:
                print(f"Batch number {i}\tloss {loss.item()}")

if __name__ == '__main__':
    dataset = RandomDataDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    train_model(dataloader)