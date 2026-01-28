"""Simple MLP network for toy environments."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_dqn_network import BaseDQNNetwork


class SimpleMLP(BaseDQNNetwork):
    """
    Simple MLP for testing DQN on toy environments.
    
    Architecture: input → hidden → hidden → output
    No fancy stuff - just linear layers with ReLU.
    """
    
    def __init__(
        self,
        input_size: int,
        num_actions: int,
        hidden_size: int = 64,
        dueling: bool = False,
    ):
        """
        Initialize Simple MLP.
        
        Args:
            input_size: Size of input observation (e.g., chain length).
            num_actions: Number of discrete actions.
            hidden_size: Size of hidden layers.
            dueling: Whether to use dueling architecture.
        """
        super().__init__(num_actions=num_actions, dueling=dueling)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if dueling:
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            
            # Value stream
            self.value_fc = nn.Linear(hidden_size, hidden_size // 2)
            self.value_out = nn.Linear(hidden_size // 2, 1)
            
            # Advantage stream
            self.adv_fc = nn.Linear(hidden_size, hidden_size // 2)
            self.adv_out = nn.Linear(hidden_size // 2, num_actions)
        else:
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_actions)
    
    def forward(
        self,
        x: torch.Tensor,
        legal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, input_size).
            legal_mask: Optional boolean mask of legal actions.
            
        Returns:
            Q-values tensor of shape (batch, num_actions).
        """
        # Flatten if needed (e.g., if input is (batch, 1, input_size))
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        if self._dueling:
            # Value stream
            v = F.relu(self.value_fc(x))
            value = self.value_out(v)
            
            # Advantage stream
            a = F.relu(self.adv_fc(x))
            advantage = self.adv_out(a)
            
            # Combine: Q = V + (A - mean(A))
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.fc3(x)
        
        return q_values
    
    def get_constructor_kwargs(self) -> Dict[str, Any]:
        """Return kwargs for serialization."""
        return {
            "input_size": self.input_size,
            "num_actions": self._num_actions,
            "hidden_size": self.hidden_size,
            "dueling": self._dueling,
        }
