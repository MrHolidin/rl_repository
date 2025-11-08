"""DQN neural network architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network for Connect Four.
    
    Uses convolutional layers to process board state.
    """

    def __init__(self, rows: int = 6, cols: int = 7, in_channels: int = 3, num_actions: int = 7):
        """
        Initialize DQN network.
        
        Args:
            rows: Number of rows in board
            cols: Number of columns in board
            in_channels: Number of input channels (3: current player, opponent, turn indicator)
            num_actions: Number of possible actions (columns)
        """
        super(DQN, self).__init__()
        
        self.rows = rows
        self.cols = cols
        self.in_channels = in_channels
        self.num_actions = num_actions
        
        # Convolutional layers
        # Use kernel_size=3 with padding=1 to preserve spatial dimensions
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Calculate flattened size dynamically
        # After conv1 with padding=1: (64, rows, cols) - size preserved
        # After conv2 with padding=1: (128, rows, cols) - size preserved
        self.flatten_size = 128 * rows * cols
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, in_channels, rows, cols)
            
        Returns:
            Q-values for each action, shape (batch, num_actions)
        """
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

