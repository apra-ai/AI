import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score

# Load/Download MNIST-Data and transform into tensor
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Make batches out of MNIST-data
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        # Create Hidden Layers and Output Layers
        # the input data is a vektor with the length of 784(28pixels*28pixels)
        # the output is a number 0 to 9 (10 digits)
        self.hl1 = nn.Linear(784, 512)
        self.hl2 = nn.Linear(512, 128)
        self.ol = nn.Linear(128, 10)

    def forward(self, x):
        # Flatten the 28*28 Pixel Matrix to a vektor
        x = x.view(-1, 784)
        
        # Go threw the layers and use activation funktions
        x = torch.relu(self.hl1(x)) #RELU funktion
        x = torch.relu(self.hl2(x)) #RELU funktion
        x = self.ol(x)              #Identity function
        return x

# initialize model
model = FNN()

# Loss funktion CrossEntropy for Classifikation and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number epochs/max iterations
epochs = 15

# Loop to train model
for epoch in range(epochs):
    #loop to train model on each batch
    for images, labels in train_loader:
        # Set gradient zero
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(images)
        
        # calculate loss
        loss = criterion(outputs, labels)
        
        # Backpropagation and optimisation
        loss.backward()
        optimizer.step()

# Calculate Accuracy
all_preds = []
all_labels = []
torch.no_grad()
for images, labels in test_loader:
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    all_preds.extend(predicted.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())
accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy * 100:.2f}%")