import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np

from WaveletKAN import WaveletKANLayer


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.layer1 = nn.Linear(784, 128)  # Flatten 28x28 images to 784 dimensions
        self.waveletLayer = WaveletKANLayer(128, 64, 10)
        self.layer2 = nn.Linear(64, 10)  # 10 classes for MNIST

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layer1(x)
        x = self.waveletLayer(x)
        x = self.layer2(x)
        return x


class MNISTNet_MLP(nn.Module):
    def __init__(self):
        super(MNISTNet_MLP, self).__init__()
        self.layer1 = nn.Linear(784, 128)  # Flatten 28x28 images to 784 dimensions
        self.MLP = nn.Linear(
            128,
            64,
        )
        self.layer2 = nn.Linear(64, 10)  # 10 classes for MNIST

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.layer1(x)
        x = torch.relu(self.MLP(x))
        x = self.layer2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST data
train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor()
)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Initialize the network and optimizer
model = MNISTNet()
# model = MNISTNet_MLP()

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()


# Training loop
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )


# Test loop
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n"
    )


# Run the training and testing
for epoch in range(1, 2):  # Train for 5 epochs
    train(epoch)
    test()
