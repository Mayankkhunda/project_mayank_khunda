import torch
import torch.nn as nn
from config import num_classes


class FlowerCNN(nn.Module):
    def __init__(self):
        super(FlowerCNN, self).__init__()

        # Convolution layers (feature extraction)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (3 → 32)
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128 → 64

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (32 → 64)
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 → 32

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (64 → 128)
            nn.ReLU(),
            nn.MaxPool2d(2)   # 32 → 16
        )

        # Fully connected layers (classification)
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # convert 3D → 1D
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Pass through convolution layers
        x = self.conv_layers(x)

        # Pass through fully connected layers
        x = self.fc_layers(x)

        return x
