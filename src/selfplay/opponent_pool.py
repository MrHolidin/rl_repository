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
    last_used: int = 0  # шаг/эпизод последнего использования


@dataclass
class SelfPlayConfig:
    """Configuration options for self-play opponent sampling."""

    start_episode: int = 0
    fraction: float = 0.5
    max_frozen_agents: int = 10
    save_every: int = 1000

    def __post_init__(self) -> None:
        if self.start_episode < 0:
            raise ValueError("start_episode must be non-negative")
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError("fraction must be between 0.0 and 1.0")
        if self.max_frozen_agents < 0:
            raise ValueError("max_frozen_agents must be non-negative")
        if self.save_every <= 0:
            raise ValueError("save_every must be positive")


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
        self_play_config: Optional[SelfPlayConfig],
        heuristic_distribution: Dict[str, float],
    ):
        """
        Initialize opponent pool.
        
        Args:
            device: Device to use for DQN agents ('cuda' or 'cpu')
            seed: Random seed
            self_play_config: Optional self-play configuration. If None, only heuristic opponents are used.
            heuristic_distribution: Distribution of heuristic opponents (will be normalized)
        """
        self.device = device
        self.seed = seed
        
        # Self-play settings
        self.self_play_config = self_play_config
        self.max_loaded_agents = (
            self_play_config.max_frozen_agents if self_play_config is not None else 0
        )
        
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
        # больше НЕ режем список по длине — ограничиваем только число loaded_agent
    
    def _sample_frozen_info(self) -> Optional[FrozenAgentInfo]:
        """Sample a random frozen agent from the pool."""
        if not self.frozen_agents:
            return None
        # выбираем из ВСЕХ чекпоинтов случайно
        return random.choice(self.frozen_agents)
    
    def _get_loaded_frozen_agent(self, info: FrozenAgentInfo, step: int) -> DQNAgent:
        """
        Lazy loading с ограничением числа моделей в памяти (LRU).
        
        Args:
            info: Frozen agent info
            step: Current episode/step number (for LRU tracking)
            
        Returns:
            Loaded DQN agent in eval mode
        """
        # Если агент уже загружен – просто вернуть и обновить last_used
        if info.loaded_agent is not None:
            info.last_used = step
            return info.loaded_agent
        
        # Сколько уже загружено
        loaded_infos = [fa for fa in self.frozen_agents if fa.loaded_agent is not None]
        if self.max_loaded_agents > 0 and len(loaded_infos) >= self.max_loaded_agents:
            # Выберем жертву: самый давно использованный
            victim = min(loaded_infos, key=lambda fa: fa.last_used)
            # Выгружаем из памяти
            victim.loaded_agent = None
        
        # Auto-detect network_type from checkpoint
        network_type = DQNAgent.get_network_type_from_checkpoint(info.checkpoint_path)
        
        # Создаём новый агент и грузим чекпоинт
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
            network_type=network_type,
        )
        
        # Load checkpoint
        agent.load(info.checkpoint_path)
        agent.eval()
        agent.epsilon = 0.0  # Ensure pure exploitation
        
        info.loaded_agent = agent
        info.last_used = step
        
        return agent
    
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
        config = self.self_play_config
        use_self_play = (
            config is not None
            and episode >= config.start_episode
            and len(self.frozen_agents) > 0
        )
        
        if use_self_play and random.random() < config.fraction:
            # Use self-play: sample a frozen agent
            info = self._sample_frozen_info()
            if info is not None:
                # episode просто используем как "step" для last_used
                return self._get_loaded_frozen_agent(info, step=episode)
        
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
            "self_play_enabled": self.self_play_config is not None,
            "self_play_ready": (
                self.self_play_config is not None 
                and len(self.frozen_agents) > 0
            ),
        }

