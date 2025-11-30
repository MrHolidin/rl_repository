"""DQN neural network architecture."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dueling_utils import dueling_aggregate


class DQN(nn.Module):
    """
    Deep Q-Network for Connect Four.
    
    Supports both standard and dueling architectures:
    - Standard: Direct Q-value estimation
    - Dueling: Separates value and advantage streams for better learning
    
    Uses Double DQN algorithm (via target network) to reduce overestimation bias.
    """

    def __init__(
        self,
        rows: int = 6,
        cols: int = 7,
        in_channels: int = 3,
        num_actions: int = 7,
        dueling: bool = False,
    ):
        super().__init__()

        self.rows = rows
        self.cols = cols
        self.in_channels = in_channels
        self.num_actions = num_actions
        self.dueling = dueling

        # --- Сверточный блок ---
        # 3×6×7 → 64×6×7
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # 64×6×7 → 128×6×7
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Ещё один conv на 128 каналах + residual (128→128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Размер после conv-блока
        self.flatten_size = 128 * rows * cols  # 128 * 6 * 7 = 5376

        # --- Полносвязные слои ---
        if dueling:
            # Общий слой перед разветвлением value/advantage
            self.fc_shared = nn.Linear(self.flatten_size, 256)
            self.shared_norm = nn.LayerNorm(256)
            self.dropout = nn.Dropout(p=0.05)

            # Value stream: V(s)
            self.fc_value_hidden = nn.Linear(256, 128)
            self.fc_value = nn.Linear(128, 1)

            # Advantage stream: A(s, a)
            self.fc_adv_hidden = nn.Linear(256, 128)
            self.fc_advantage = nn.Linear(128, num_actions)
        else:
            # Стандартный DQN head
            self.fc1 = nn.Linear(self.flatten_size, 256)
            self.fc2 = nn.Linear(256, num_actions)

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, rows, cols)
            legal_mask: Optional mask of legal actions for dueling aggregation
        Returns:
            Q-values: (batch, num_actions)
        """

        # --- Conv-блок с residual на уровне feature maps ---
        x = F.relu(self.bn1(self.conv1(x)))        # (B, 64, H, W)
        x = F.relu(self.bn2(self.conv2(x)))        # (B, 128, H, W)

        # Residual: conv3 + skip из выхода conv2
        residual = x
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x + residual)                   # (B, 128, H, W)

        # --- Flatten ---
        x = x.view(x.size(0), -1)                  # (B, 128*rows*cols)

        if self.dueling:
            # Общий слой
            x = self.fc_shared(x)
            x = self.shared_norm(x)
            x = F.relu(x)
            x = self.dropout(x)

            # Value stream
            value = F.relu(self.fc_value_hidden(x))    # (B, 128)
            value = self.fc_value(value)               # (B, 1)

            # Advantage stream
            advantage = F.relu(self.fc_adv_hidden(x))  # (B, 128)
            advantage = self.fc_advantage(advantage)   # (B, num_actions)

            q_values = dueling_aggregate(value, advantage, legal_mask)
        else:
            # Обычный DQN head
            x = F.relu(self.fc1(x))
            q_values = self.fc2(x)

        return q_values
