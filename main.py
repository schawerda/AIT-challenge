# AIT challenge
# Separate waste efficently with deep learning

# Import packages
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from efficientnet_pytorch import EfficientNet
from tqdm import tqdm

# Custom dataset class
from utils import ImageDataset

data_dir = 'data'

if __name__ == '__main__':

    # Load the dataset
    data_train = ImageDataset(data_dir, dataset_type='train')
    data_val   = ImageDataset(data_dir, dataset_type='val')

    train_loader = DataLoader(data_train, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(data_val, batch_size=32, shuffle=False, num_workers=4)

    # Load the model
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, 6)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training and validation
    def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):

        # For each epoch
        for epoch in range(num_epochs):
            model.train()  # Training mode
            train_loss = 0.0
            correct = 0

            # Training
            for images, labels in tqdm(train_loader, desc="Training", leave=False):
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

            # Print current results
            train_accuracy = correct / len(train_loader.dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss/len(train_loader.dataset):.4f}, Accuracy: {train_accuracy:.4f}")

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0

            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc="Validation", leave=False):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()

            # Print current validation results
            val_accuracy = correct / len(val_loader.dataset)
            print(f"Validation Loss: {val_loss/len(val_loader.dataset):.4f}, Validation Accuracy: {val_accuracy:.4f}")

        # Save the model
        torch.save(model.state_dict(), "classifier.pth")

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
