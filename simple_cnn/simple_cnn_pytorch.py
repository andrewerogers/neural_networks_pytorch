import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728),   # SVHN mean per channel
                         (0.1980, 0.2010, 0.1970))   # SVHN std per channel
])

# Load SVHN - note the 'split' parameter instead of 'train'
train_dataset = datasets.SVHN(root='./data', split='train', 
                              download=True, transform=transform)
test_dataset = datasets.SVHN(root='./data', split='test', 
                             download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class SimpleCNN(nn.Module):
    """
    A convolutional neural network for MNIST digit classification.
    
    Architecture:
        Input (28×28 grayscale) 
        → Conv2D (1→32 channels, 3×3 kernel, padding=1)
        → ReLU activation
        → MaxPool2D (2×2, stride=2)
        → Conv2D (32→64 channels, 3×3 kernel, padding=1)
        → ReLU activation
        → MaxPool2D (2×2, stride=2)
        → Flatten
        → Fully Connected (6400 → 128)
        → ReLU activation
        → Fully Connected (128 → 10)
        → Output logits
    """
    
    def __init__(self, input_channels=1, num_classes=10):
        """
        Initialize the MNIST CNN model.
        
        Args:
            input_channels (int): Number of input channels. Default=1 for grayscale images.
            num_classes (int): Number of output classes. Default=10 for digits 0-9.
        """
        super(SimpleCNN, self).__init__()
        
        # First Convolutional Block
        # Input: [batch, 1, 28, 28]
        # Conv2D: in_channels=1 (grayscale), out_channels=32 filters, kernel=3×3
        # padding=1 keeps spatial dimensions: 28×28 → 28×28
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=3,
            padding=1,
            bias=True
        )
        # Output: [batch, 32, 28, 28]
        
        # First Activation Function
        # ReLU(x) = max(0, x) - introduces non-linearity
        self.relu1 = nn.ReLU(inplace=True)
        # inplace=True saves memory by modifying tensor in-place
        
        # First Max Pooling Layer
        # Reduces spatial dimensions by 2× in each direction
        # kernel_size=2 with stride=2 means no overlap
        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        # Output: [batch, 32, 14, 14] (28/2 = 14)
        
        # Second Convolutional Block
        # Now taking 32 input channels and producing 64 feature maps
        # Deeper layers capture more complex patterns
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=True
        )
        # Output: [batch, 64, 14, 14]
        
        # Second Activation Function
        self.relu2 = nn.ReLU(inplace=True)
        
        # Second Max Pooling Layer
        # Reduces 14×14 → 7×7
        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        # Output: [batch, 64, 7, 7]
        
        # Flatten Layer
        # Converts 4D tensor [batch, channels, height, width] to 2D [batch, features]
        # No learnable parameters, just a reshape operation
        self.flatten = nn.Flatten()
        # Flattened size: batch × (64 × 7 × 7) = batch × 3136
        
        # Fully Connected (Dense) Layers for Classification
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        # Output: [batch, 128]
        
        # Activation on hidden layer
        self.relu3 = nn.ReLU(inplace=True)
        
        # Optional: Dropout for regularization (helps prevent overfitting)
        # Randomly zeros 20% of activations during training
        # This forces the network to learn redundant representations
        self.dropout = nn.Dropout(p=0.2)
        
        # Output layer: 128 hidden units → 10 class logits
        # (logits are raw, un-softmaxed scores)
        self.fc2 = nn.Linear(128, num_classes)
        # Output: [batch, 10]
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 1, 28, 28]
        
        Returns:
            torch.Tensor: Output logits of shape [batch_size, 10]
        """
        # First convolutional block
        x = self.conv1(x)      # [B, 1, 28, 28] → [B, 32, 28, 28]
        x = self.relu1(x)      # Apply non-linearity
        x = self.pool1(x)      # [B, 32, 28, 28] → [B, 32, 14, 14]
        
        # Second convolutional block
        x = self.conv2(x)      # [B, 32, 14, 14] → [B, 64, 14, 14]
        x = self.relu2(x)      # Apply non-linearity
        x = self.pool2(x)      # [B, 64, 14, 14] → [B, 64, 7, 7]
        
        # Fully connected layers
        x = self.flatten(x)    # [B, 64, 7, 7] → [B, 3136]
        x = self.fc1(x)        # [B, 3136] → [B, 128]
        x = self.relu3(x)      # Apply non-linearity to hidden layer
        x = self.dropout(x)    # Randomly zero 20% of activations (only during training)
        x = self.fc2(x)        # [B, 128] → [B, 10] (logits)
        
        return x


# Initialize model, loss, optimizer
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using PyTorch device: {device}\n')
model = SimpleCNN(input_channels=3, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()  # Combines softmax + negative log likelihood
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return total_loss / len(train_loader), accuracy


def evaluate(model, test_loader, criterion, device):
    model.eval()  # Set to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation (saves memory, faster)
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Count correct predictions
            _, predicted = torch.max(outputs.data, 1)  # Get argmax
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return total_loss / len(test_loader), accuracy

# Training loop
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return total_loss / len(train_loader), accuracy

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%\n")