"""Chain MDP - simple environment for testing DQN."""

from typing import Optional

import numpy as np

from ..base import StepResult


class ChainMDP:
    """
    Simple chain environment: s0 → s1 → s2 → ... → sN-1 (terminal, reward=1)
    
    Actions:
        0: stay in place
        1: move right
    
    This is useful for testing:
        - Bootstrap works correctly (Q should propagate back)
        - Q-values don't explode (max Q should be ~γ^(N-1-state))
        - Network can learn simple patterns
    
    Expected Q-values for optimal policy (always go right):
        Q(s_i, right) = γ^(N-1-i)
        Q(s_i, stay) < Q(s_i, right)
    
    Example with length=5, γ=0.99:
        Q(s0) ≈ 0.96, Q(s1) ≈ 0.97, Q(s2) ≈ 0.98, Q(s3) ≈ 0.99, Q(s4) = 1.0
    """
    
    def __init__(self, length: int = 5, max_steps: int = 100):
        """
        Initialize Chain MDP.
        
        Args:
            length: Number of states (including terminal state).
            max_steps: Maximum steps before truncation.
        """
        self.length = length
        self.max_steps = max_steps
        self.state = 0
        self.steps = 0
        self._num_actions = 2
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset to initial state."""
        if seed is not None:
            np.random.seed(seed)
        self.state = 0
        self.steps = 0
        return self._get_obs()
    
    def step(self, action: int) -> StepResult:
        """
        Take action in environment.
        
        Args:
            action: 0 = stay, 1 = move right
            
        Returns:
            StepResult with observation, reward, done flags, info.
        """
        self.steps += 1
        
        # Action 1 moves right, action 0 stays
        if action == 1 and self.state < self.length - 1:
            self.state += 1
        
        # Check if reached terminal state
        terminated = (self.state == self.length - 1)
        truncated = (self.steps >= self.max_steps) and not terminated
        
        reward = 1.0 if terminated else 0.0
        
        return StepResult(
            obs=self._get_obs(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={"state": self.state, "steps": self.steps},
        )
    
    @property
    def legal_actions_mask(self) -> np.ndarray:
        """All actions are always legal."""
        return np.ones(self._num_actions, dtype=bool)
    
    @property
    def num_actions(self) -> int:
        """Number of available actions."""
        return self._num_actions
    
    @property
    def observation_shape(self) -> tuple:
        """Shape of observation."""
        return (self.length,)
    
    def _get_obs(self) -> np.ndarray:
        """Get one-hot encoded observation of current state."""
        obs = np.zeros(self.length, dtype=np.float32)
        obs[self.state] = 1.0
        return obs
    
    def render(self) -> None:
        """Print current state."""
        chain = ["_"] * self.length
        chain[self.state] = "X"
        chain[-1] = "G" if self.state != self.length - 1 else "X"
        print(f"[{' '.join(chain)}] step={self.steps}")
