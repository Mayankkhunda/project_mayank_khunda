import torch
import torch.nn as nn
import torch.optim as optim

from model import FlowerCNN
from dataset import flowerLoader
from config import epochs, learning_rate, data_path, model_save_path


def train_model(model, num_epochs, train_loader, loss_fn, optimizer):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for images, labels in train_loader:
            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = loss_fn(outputs, labels)

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss:.4f}")

    # Save trained model
    torch.save(model.state_dict(), model_save_path)
    print(" Model saved at:", model_save_path)


#  This block allows direct running of train.py
if __name__ == "__main__":
    print(" Starting training...")

    # Load data
    train_loader = flowerLoader(data_path)

    # Initialize model
    model = FlowerCNN()

    # Define loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model
    train_model(model, epochs, train_loader, loss_fn, optimizer)

    print(" Training completed!")
