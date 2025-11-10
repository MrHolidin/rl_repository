"""DQN neural network architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network for Connect Four.
    
    Supports both standard and dueling architectures:
    - Standard: Direct Q-value estimation
    - Dueling: Separates value and advantage streams for better learning
    
    Uses Double DQN algorithm (via target network) to reduce overestimation bias.
    """

    def __init__(self, rows: int = 6, cols: int = 7, in_channels: int = 3, num_actions: int = 7, dueling: bool = False):
        """
        Initialize DQN network.
        
        Args:
            rows: Number of rows in board
            cols: Number of columns in board
            in_channels: Number of input channels (3: current player, opponent, turn indicator)
            num_actions: Number of possible actions (columns)
            dueling: If True, use dueling architecture (separates value and advantage streams)
        """
        super(DQN, self).__init__()
        
        self.rows = rows
        self.cols = cols
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.dueling = dueling
        
        # Convolutional layers (shared for both architectures)
        # Use kernel_size=3 with padding=1 to preserve spatial dimensions
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Calculate flattened size dynamically
        # After conv1 with padding=1: (64, rows, cols) - size preserved
        # After conv2 with padding=1: (128, rows, cols) - size preserved
        self.flatten_size = 128 * rows * cols
        
        if dueling:
            # Dueling architecture: separate value and advantage streams
            # Shared fully connected layer
            self.fc_shared = nn.Linear(self.flatten_size, 256)
            
            # Value stream: estimates V(s)
            self.fc_value = nn.Linear(256, 1)
            
            # Advantage stream: estimates A(s, a)
            self.fc_advantage = nn.Linear(256, num_actions)
        else:
            # Standard architecture: direct Q-value estimation
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
        # Shared convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        if self.dueling:
            # Dueling architecture
            # Shared fully connected layer
            x = F.relu(self.fc_shared(x))
            
            # Value stream: V(s)
            value = self.fc_value(x)  # (batch, 1)
            
            # Advantage stream: A(s, a)
            advantage = self.fc_advantage(x)  # (batch, num_actions)
            
            # Combine: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
            # This ensures that the advantage stream learns relative advantages
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            # Standard architecture
            x = F.relu(self.fc1(x))
            q_values = self.fc2(x)
        
        return q_values

