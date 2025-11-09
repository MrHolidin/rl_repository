"""Opponent pool for self-play training."""

import os
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from src.agents import DQNAgent, RandomAgent, HeuristicAgent, SmartHeuristicAgent


@dataclass
class FrozenAgentInfo:
    """Information about a frozen agent checkpoint."""
    checkpoint_path: str
    episode: int
    loaded_agent: Optional[DQNAgent] = None  # lazy loading


class OpponentPool:
    """
    Pool of opponents for self-play training.
    
    Manages both heuristic opponents and frozen DQN agents from previous checkpoints.
    Supports lazy loading to avoid keeping all models in memory.
    """
    
    def __init__(
        self,
        device: Optional[str],
        seed: int,
        self_play_enabled: bool,
        self_play_start_episode: int,
        self_play_fraction: float,
        max_frozen_agents: int,
        heuristic_distribution: Dict[str, float],
    ):
        """
        Initialize opponent pool.
        
        Args:
            device: Device to use for DQN agents ('cuda' or 'cpu')
            seed: Random seed
            self_play_enabled: Whether self-play is enabled
            self_play_start_episode: Episode number to start self-play
            self_play_fraction: Fraction of episodes to use self-play (0.0 to 1.0)
            max_frozen_agents: Maximum number of frozen agents to keep in pool
            heuristic_distribution: Distribution of heuristic opponents (will be normalized)
        """
        self.device = device
        self.seed = seed
        
        # Self-play settings
        self.self_play_enabled = self_play_enabled
        self.self_play_start_episode = self_play_start_episode
        self.self_play_fraction = self_play_fraction
        self.max_frozen_agents = max_frozen_agents
        
        # Normalize heuristic distribution
        total = sum(heuristic_distribution.values())
        if total > 0:
            self.heuristic_distribution = {
                k: v / total for k, v in heuristic_distribution.items()
            }
        else:
            # Default distribution if empty
            self.heuristic_distribution = {
                "random": 0.2,
                "heuristic": 0.5,
                "smart_heuristic": 0.3,
            }
        
        # List of frozen agents
        self.frozen_agents: List[FrozenAgentInfo] = []
    
    # ---------- HEURISTIC OPPONENTS ----------
    
    def _sample_heuristic_type(self) -> str:
        """Sample a heuristic opponent type based on distribution."""
        r = random.random()
        acc = 0.0
        for name, p in self.heuristic_distribution.items():
            acc += p
            if r <= acc:
                return name
        return list(self.heuristic_distribution.keys())[-1]
    
    def _create_heuristic_agent(self, episode: int):
        """
        Create a new heuristic agent with seed depending on episode.
        
        Args:
            episode: Current episode number (for seed variation)
            
        Returns:
            Heuristic agent instance
        """
        opp_type = self._sample_heuristic_type()
        opp_seed = self.seed + 100000 + episode  # Avoid seed collisions
        
        if opp_type == "random":
            return RandomAgent(seed=opp_seed)
        elif opp_type == "heuristic":
            return HeuristicAgent(seed=opp_seed)
        elif opp_type == "smart_heuristic":
            return SmartHeuristicAgent(seed=opp_seed)
        else:
            raise ValueError(f"Unknown heuristic opponent type: {opp_type}")
    
    # ---------- FROZEN DQN AGENTS ----------
    
    def add_frozen_agent(self, checkpoint_path: str, episode: int):
        """
        Add a new frozen agent to the pool.
        
        Args:
            checkpoint_path: Path to checkpoint file
            episode: Episode number when this checkpoint was saved
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        info = FrozenAgentInfo(checkpoint_path=checkpoint_path, episode=episode)
        self.frozen_agents.append(info)
        
        # If we exceeded the limit, keep only the most recent agents
        if len(self.frozen_agents) > self.max_frozen_agents:
            # Sort by episode and keep the most recent ones
            self.frozen_agents.sort(key=lambda x: x.episode)
            self.frozen_agents = self.frozen_agents[-self.max_frozen_agents:]
    
    def _sample_frozen_info(self) -> Optional[FrozenAgentInfo]:
        """Sample a random frozen agent from the pool."""
        if not self.frozen_agents:
            return None
        return random.choice(self.frozen_agents)
    
    def _get_loaded_frozen_agent(self, info: FrozenAgentInfo) -> DQNAgent:
        """
        Lazy loading: load agent from disk only when first used.
        
        Args:
            info: Frozen agent info
            
        Returns:
            Loaded DQN agent in eval mode
        """
        if info.loaded_agent is None:
            # Create agent with default hyperparameters
            # We'll load the state_dict anyway, so these don't matter much
            agent = DQNAgent(
                rows=6,
                cols=7,
                learning_rate=0.001,  # Not used in eval mode
                discount_factor=0.99,
                epsilon=0.0,  # Pure exploitation
                epsilon_decay=0.995,
                epsilon_min=0.0,
                batch_size=32,
                replay_buffer_size=1000,  # Minimal, not used in eval
                target_update_freq=100,
                soft_update=False,
                tau=0.01,
                device=self.device,
                seed=self.seed,
            )
            
            # Load checkpoint
            agent.load(info.checkpoint_path)
            agent.eval()
            agent.epsilon = 0.0  # Ensure pure exploitation
            
            info.loaded_agent = agent
        
        return info.loaded_agent
    
    # ---------- PUBLIC INTERFACE ----------
    
    def sample_heuristic_opponent(self, episode: int):
        """
        Sample a heuristic opponent (ignores self-play, always returns heuristic).
        
        Args:
            episode: Current episode number (for seed variation)
            
        Returns:
            Heuristic agent instance
        """
        return self._create_heuristic_agent(episode)
    
    def sample_opponent(self, episode: int):
        """
        Sample an opponent for the given episode.
        
        Args:
            episode: Current episode number
            
        Returns:
            Agent instance (heuristic or frozen DQN)
        """
        use_self_play = (
            self.self_play_enabled
            and episode >= self.self_play_start_episode
            and len(self.frozen_agents) > 0
        )
        
        if use_self_play and random.random() < self.self_play_fraction:
            # Use self-play: sample a frozen agent
            info = self._sample_frozen_info()
            if info is not None:
                return self._get_loaded_frozen_agent(info)
        
        # Fallback: use heuristic opponent
        return self._create_heuristic_agent(episode)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the opponent pool.
        
        Returns:
            Dictionary with pool statistics
        """
        loaded_count = sum(1 for info in self.frozen_agents if info.loaded_agent is not None)
        
        return {
            "frozen_agents_count": len(self.frozen_agents),
            "loaded_agents_count": loaded_count,
            "self_play_enabled": self.self_play_enabled,
            "self_play_ready": (
                self.self_play_enabled 
                and len(self.frozen_agents) > 0
            ),
        }

