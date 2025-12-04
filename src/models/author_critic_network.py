"""Actor-Critic network for PPO (shared conv trunk with DQN)."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticCNN(nn.Module):
    """
    Actor-Critic сеть для Connect Four.
    
    Общий сверточный блок как в DQN, дальше:
    - policy_head: logits по действиям
    - value_head: скаляр V(s)
    """

    def __init__(
        self,
        rows: int = 6,
        cols: int = 7,
        in_channels: int = 3,
        num_actions: int = 7,
    ):
        super().__init__()

        self.rows = rows
        self.cols = cols
        self.in_channels = in_channels
        self.num_actions = num_actions

        # --- Conv-блок такой же, как в DQN ---
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.flatten_size = 128 * rows * cols  # 128 * 6 * 7 = 5376

        # Общий fully connected слой
        self.fc_shared = nn.Linear(self.flatten_size, 256)
        self.shared_norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(p=0.00)

        # Policy head: logits по действиям
        self.policy_head = nn.Linear(256, num_actions)

        # Value head: V(s)
        self.value_head = nn.Linear(256, 1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Общий encoder для feature maps."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        residual = x
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x + residual)

        x = x.view(x.size(0), -1)
        x = self.fc_shared(x)
        x = self.shared_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, in_channels, rows, cols)
            legal_mask: (batch, num_actions) bool, опционально

        Returns:
            logits: (batch, num_actions)
            values: (batch,)  — V(s)
        """
        h = self._encode(x)

        logits = self.policy_head(h)      # (B, A)
        values = self.value_head(h).squeeze(-1)  # (B,)

        if legal_mask is not None:
            # Маскируем нелегальные действия
            # ожидаем legal_mask.dtype == bool
            logits = logits.masked_fill(~legal_mask, float('-inf'))

        return logits, values
