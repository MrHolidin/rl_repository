"""AlphaZero agent implementation."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..base_agent import BaseAgent
from .replay_buffer import AlphaZeroBatch, AlphaZeroReplayBuffer


class AlphaZeroAgent(BaseAgent):
    """
    AlphaZero agent with dual-head network.

    Trains on (state, pi_mcts, z) tuples with policy + value loss.
    """

    def __init__(
        self,
        network: nn.Module,
        num_actions: int,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        replay_buffer_size: int = 100_000,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        value_loss_weight: float = 1.0,
        grad_clip_norm: float = 1.0,
    ):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.value_loss_weight = value_loss_weight
        self.grad_clip_norm = grad_clip_norm
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.replay_buffer_size = replay_buffer_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.network = network.to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.replay_buffer = AlphaZeroReplayBuffer(
            capacity=replay_buffer_size,
            seed=seed,
        )

        self._train_steps = 0
        self._training = True

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

    def act(
        self,
        obs: np.ndarray,
        legal_mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> int:
        """Select action using network policy (no MCTS)."""
        if legal_mask is None:
            legal_mask = np.ones(self.num_actions, dtype=bool)

        with torch.no_grad():
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            mask_t = torch.as_tensor(
                legal_mask, dtype=torch.bool, device=self.device
            ).unsqueeze(0)

            policy_probs, _ = self.network.predict(obs_t, mask_t)
            policy_probs = policy_probs[0].cpu().numpy()

        if deterministic:
            return int(np.argmax(policy_probs))
        else:
            policy_probs = np.maximum(policy_probs, 0)
            total = policy_probs.sum()
            if total > 0:
                policy_probs = policy_probs / total
            else:
                legal_indices = np.where(legal_mask)[0]
                return int(np.random.choice(legal_indices))
            return int(np.random.choice(self.num_actions, p=policy_probs))

    def predict(
        self,
        obs: np.ndarray,
        legal_mask: np.ndarray,
    ) -> Tuple[Dict[int, float], float]:
        """Get policy distribution and value for MCTS."""
        with torch.inference_mode():
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            mask_t = torch.as_tensor(
                legal_mask, dtype=torch.bool, device=self.device
            ).unsqueeze(0)

            policy_probs, value = self.network.predict(obs_t, mask_t)
            policy_probs = policy_probs[0].cpu().numpy()
            value = value[0, 0].item()

        policy_dict = {
            a: float(policy_probs[a]) for a in range(self.num_actions) if legal_mask[a]
        }
        return policy_dict, value

    def compile_network(self, mode: str = "reduce-overhead", warmup_batch_sizes: list = None) -> None:
        """
        Compile network with torch.compile for faster inference.
        
        Args:
            mode: Compilation mode ('reduce-overhead' recommended for inference)
            warmup_batch_sizes: List of batch sizes to warmup (triggers JIT for each)
        """
        if not hasattr(torch, "compile"):
            return
            
        self.network = torch.compile(self.network, mode=mode)
        
        if warmup_batch_sizes:
            print(f"Warming up torch.compile for batch sizes: {warmup_batch_sizes}")
            for bs in warmup_batch_sizes:
                dummy_obs = torch.randn(bs, 2, 6, 7, device=self.device)
                dummy_mask = torch.ones(bs, 7, dtype=torch.bool, device=self.device)
                with torch.inference_mode():
                    _ = self.network.predict(dummy_obs, dummy_mask)
            torch.cuda.synchronize()
            print("Warmup complete")

    def train_on_batch(self, batch: AlphaZeroBatch) -> Dict[str, float]:
        """Train on a batch of samples."""
        self.network.train()

        policy_logits, value_pred = self.network(
            batch.observations,
            legal_mask=None,
        )

        masked_logits = policy_logits.masked_fill(~batch.legal_masks, -1e9)
        log_probs = torch.log_softmax(masked_logits, dim=-1)

        valid_mask = batch.target_policies > 0
        policy_loss = -(batch.target_policies * log_probs)
        policy_loss = policy_loss.masked_fill(~valid_mask, 0.0).sum(dim=-1).mean()

        value_loss = nn.functional.mse_loss(value_pred, batch.target_values)

        loss = policy_loss + self.value_loss_weight * value_loss

        self.optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            self.grad_clip_norm,
        )

        self.optimizer.step()
        self._train_steps += 1

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "grad_norm": grad_norm.item(),
        }

    def update(self) -> Dict[str, float]:
        """Perform one training step if buffer has enough samples."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        if not self._training:
            return {}

        batch = self.replay_buffer.sample(self.batch_size, self.device)
        return self.train_on_batch(batch)

    def train(self) -> None:
        self._training = True
        self.network.train()

    def eval(self) -> None:
        self._training = False
        self.network.eval()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        checkpoint = {
            "network_class": self.network.get_class_name(),
            "network_kwargs": self.network.get_constructor_kwargs(),
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_actions": self.num_actions,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "replay_buffer_size": self.replay_buffer_size,
            "value_loss_weight": self.value_loss_weight,
            "grad_clip_norm": self.grad_clip_norm,
            "train_steps": self._train_steps,
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(
        cls, path: str, *, device: Optional[str] = None, **overrides
    ) -> "AlphaZeroAgent":
        from src.models.alphazero import Connect4AlphaZeroNetwork

        map_location = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        network_class = checkpoint["network_class"]
        network_kwargs = checkpoint["network_kwargs"]

        if network_class == "Connect4AlphaZeroNetwork":
            network = Connect4AlphaZeroNetwork(**network_kwargs)
        else:
            raise ValueError(f"Unknown network class: {network_class}")

        agent = cls(
            network=network,
            num_actions=checkpoint["num_actions"],
            learning_rate=checkpoint.get("learning_rate", 0.001),
            weight_decay=checkpoint.get("weight_decay", 1e-4),
            batch_size=checkpoint.get("batch_size", 256),
            replay_buffer_size=checkpoint.get("replay_buffer_size", 100_000),
            value_loss_weight=checkpoint.get("value_loss_weight", 1.0),
            grad_clip_norm=checkpoint.get("grad_clip_norm", 1.0),
            device=device,
            **overrides,
        )

        agent.network.load_state_dict(checkpoint["network_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent._train_steps = checkpoint.get("train_steps", 0)

        return agent
