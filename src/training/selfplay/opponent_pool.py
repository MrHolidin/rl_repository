"""Opponent pool for self-play training."""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from src.agents import RandomAgent, HeuristicAgent, SmartHeuristicAgent
from src.agents.othello import OthelloHeuristicAgent
from src.agents.base_agent import BaseAgent
from src.envs.base import StepResult
from src.utils import freeze_agent


@dataclass
class FrozenAgentInfo:
    """Metadata for one frozen checkpoint + running stats vs the learner."""

    checkpoint_path: str
    episode: int
    loaded_agent: Optional[BaseAgent] = None  # lazy loading (DQN / PPO / …)
    # EMA of P(frozen wins the match). Starts at 0.5; updated once per game vs this frozen.
    ema_win_rate: float = 0.5
    last_used: int = 0  # шаг/эпизод последнего использования
    games: int = 0
    wins: int = 0  # frozen wins (learner lost)
    losses: int = 0  # frozen losses (learner won)
    draws: int = 0

    @property
    def cumulative_win_rate(self) -> float:
        if self.games <= 0:
            return 0.5
        return self.wins / self.games

    @property
    def win_rate(self) -> float:
        """Sampling weight input: recent frozen strength (EMA), not lifetime ratio."""
        return self.ema_win_rate


@dataclass
class SelfPlayConfig:
    """Configuration options for self-play opponent sampling."""

    start_episode: int = 0
    current_self_fraction: float = 0.3
    past_self_fraction: float = 0.3
    max_frozen_agents: int = 10
    save_every: int = 1000
    # EMA update for FrozenAgentInfo.ema_win_rate after each match vs a frozen checkpoint.
    frozen_ema_beta: float = 0.05

    def __post_init__(self) -> None:
        if self.start_episode < 0:
            raise ValueError("start_episode must be non-negative")
        if not 0.0 <= self.current_self_fraction <= 1.0:
            raise ValueError("current_self_fraction must be between 0.0 and 1.0")
        if not 0.0 <= self.past_self_fraction <= 1.0:
            raise ValueError("past_self_fraction must be between 0.0 and 1.0")
        if self.current_self_fraction + self.past_self_fraction > 1.0:
            raise ValueError("current_self_fraction + past_self_fraction must be <= 1.0")
        if self.max_frozen_agents < 0:
            raise ValueError("max_frozen_agents must be non-negative")
        if self.save_every <= 0:
            raise ValueError("save_every must be positive")
        if not 0.0 < self.frozen_ema_beta <= 1.0:
            raise ValueError("frozen_ema_beta must be in (0, 1]")


ScriptedMode = Literal["classic", "minibg"]


@dataclass(frozen=True)
class ScriptedOpponentsSpec:
    """Weights for sampling a non-learned opponent **inside the scripted branch** (after self-play rolls)."""

    mode: ScriptedMode
    distribution: Dict[str, float]

    def __post_init__(self) -> None:
        if not self.distribution:
            raise ValueError("scripted.distribution must be non-empty")
        tot = sum(max(0.0, float(v)) for v in self.distribution.values())
        if tot <= 0:
            raise ValueError("scripted.distribution must have a positive total weight")
        norm = {str(k).strip(): max(0.0, float(v)) / tot for k, v in self.distribution.items() if str(k).strip()}
        object.__setattr__(self, "distribution", norm)


class SelfPlayOpponent(BaseAgent):
    """Lightweight wrapper around the current agent for self-play."""

    def __init__(self, base_agent: BaseAgent, *, greedy: bool = True) -> None:
        self._base_agent = base_agent
        self._greedy = greedy

    def act(
        self,
        obs,
        legal_mask: Optional[Any] = None,
        deterministic: bool = False,
    ) -> int:
        use_det = deterministic or self._greedy
        base = self._base_agent
        # Greedy / deterministic play must use mean weights for NoisyLinear
        # (same as frozen checkpoints). The learner stays in train(); toggle only
        # for this forward so exploration noise does not distort the opponent.
        if use_det and getattr(base, "use_noisy_nets", False):
            was_training = base.training
            base.eval()
            try:
                return base.act(obs, legal_mask=legal_mask, deterministic=True)
            finally:
                if was_training:
                    base.train()
        return base.act(obs, legal_mask=legal_mask, deterministic=use_det)

    def opponent_step(
        self,
        env: Any,
        obs,
        *,
        legal_mask: Optional[Any] = None,
        deterministic: bool = False,
    ) -> StepResult:
        """Delegates structured opponents (MiniBG structured PPO) to ``base.opponent_step``."""
        use_det = deterministic or self._greedy
        base = self._base_agent
        if use_det and getattr(base, "use_noisy_nets", False):
            was_training = base.training
            base.eval()
            try:
                if hasattr(base, "opponent_step"):
                    return base.opponent_step(env, obs, legal_mask=legal_mask, deterministic=True)
                opp_action = base.act(obs, legal_mask=legal_mask, deterministic=True)
                return env.step(opp_action)
            finally:
                if was_training:
                    base.train()
        if hasattr(base, "opponent_step"):
            return base.opponent_step(env, obs, legal_mask=legal_mask, deterministic=use_det)
        opp_action = base.act(obs, legal_mask=legal_mask, deterministic=use_det)
        return env.step(opp_action)

    def save(self, path: str) -> None:  # pragma: no cover - not used
        return None

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "SelfPlayOpponent":  # pragma: no cover - unused
        raise NotImplementedError("SelfPlayOpponent cannot be loaded from disk.")

    def eval(self) -> None:
        return None

    def train(self) -> None:
        return None


class OpponentPool:
    """
    Pool of opponents for self-play training.

    The **scripted** branch (``ScriptedOpponentsSpec``) mixes game-specific static opponents:
    classic board presets or MiniBG bot ids, with optional ``random`` key for ``RandomAgent``.
    """

    def __init__(
        self,
        device: Optional[str],
        seed: int,
        self_play_config: Optional[SelfPlayConfig],
        scripted: ScriptedOpponentsSpec,
        current_agent: Optional[BaseAgent] = None,
    ):
        self.device = device
        self.seed = seed
        self._scripted = scripted
        if scripted.mode == "minibg":
            from src.envs.minibg.heuristic_bots.bots import default_bot_constructors

            valid = frozenset(default_bot_constructors().keys())
            for name in scripted.distribution:
                if name != "random" and name not in valid:
                    raise ValueError(
                        f"minibg scripted distribution: unknown key {name!r}; "
                        f"expected 'random' or {sorted(valid)}"
                    )

        # Self-play settings
        self.self_play_config = self_play_config
        self.max_loaded_agents = (
            self_play_config.max_frozen_agents if self_play_config is not None else 0
        )
        self.current_agent = current_agent

        # List of frozen agents
        self.frozen_agents: List[FrozenAgentInfo] = []

    # ---------- SCRIPTED (NON-LEARNED) OPPONENTS ----------

    def _sample_scripted_key(self) -> str:
        r = random.random()
        acc = 0.0
        dist = self._scripted.distribution
        for name, p in dist.items():
            acc += p
            if r <= acc:
                return name
        return list(dist.keys())[-1]

    def _create_scripted_opponent(self, episode: int):
        key = self._sample_scripted_key()
        if self._scripted.mode == "minibg":
            from src.envs.minibg.heuristic_bots.agent_adapter import MiniBGHeuristicAgent
            from src.envs.minibg.heuristic_bots.tournament import make_bot

            rng_ep = self.seed + 100000 + episode
            if key == "random":
                return RandomAgent(seed=self.seed + 200000 + episode)
            return MiniBGHeuristicAgent(make_bot(key, rng_ep + 31))

        opp_seed = self.seed + 100000 + episode
        if key == "random":
            return RandomAgent(seed=opp_seed)
        if key == "heuristic":
            return HeuristicAgent(seed=opp_seed)
        if key == "smart_heuristic":
            return SmartHeuristicAgent(seed=opp_seed)
        if key == "othello_heuristic":
            return OthelloHeuristicAgent(seed=opp_seed)
        raise ValueError(f"Unknown scripted opponent type: {key}")
    
    # ---------- FROZEN CHECKPOINT AGENTS (DQN / PPO) ----------
    
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

    def apply_episode_result(
        self,
        opponent: Optional[BaseAgent],
        agent_result: Optional[int],
    ) -> None:
        """
        After a full episode: if ``opponent`` was a frozen loaded agent, update
        cumulative W/L/D and EMA win-rate (one update per match).
        ``agent_result``: 1 = learner won, -1 = learner lost, 0 = draw.
        """
        if opponent is None or agent_result is None:
            return
        cfg = self.self_play_config
        beta = float(cfg.frozen_ema_beta) if cfg is not None else 0.05

        for info in self.frozen_agents:
            if info.loaded_agent is not opponent:
                continue
            info.games += 1
            if agent_result == 1:
                info.losses += 1
                y = 0.0
            elif agent_result == -1:
                info.wins += 1
                y = 1.0
            elif agent_result == 0:
                info.draws += 1
                y = 0.5
            else:
                return
            info.ema_win_rate = (1.0 - beta) * info.ema_win_rate + beta * y
            return

    def _sample_frozen_info(self) -> Optional[FrozenAgentInfo]:
        """Sample a frozen agent with PFSP-like weights (hard: prefer strong opponents)."""
        if not self.frozen_agents:
            return None

        weights: List[float] = []
        eps = 1e-2  # чтобы никто не выпадал совсем
        for info in self.frozen_agents:
            p = info.win_rate  
            w = max(p, eps) ** 2
            weights.append(w)

        total = sum(weights)
        if total <= 0:
            return random.choice(self.frozen_agents)

        r = random.random() * total
        acc = 0.0
        for info, weight in zip(self.frozen_agents, weights):
            acc += weight
            if r <= acc:
                return info

        return self.frozen_agents[-1]

    
    def _get_loaded_frozen_agent(self, info: FrozenAgentInfo, step: int) -> BaseAgent:
        """
        Lazy loading с ограничением числа моделей в памяти (LRU).

        Loads DQN or PPO via ``load_training_agent_checkpoint`` (same as offline eval).
        """
        if info.loaded_agent is not None:
            info.last_used = step
            return info.loaded_agent

        loaded_infos = [fa for fa in self.frozen_agents if fa.loaded_agent is not None]
        if self.max_loaded_agents > 0 and len(loaded_infos) >= self.max_loaded_agents:
            victim = min(loaded_infos, key=lambda fa: fa.last_used)
            victim.loaded_agent = None

        from src.evaluation.eval_checkpoints import load_training_agent_checkpoint

        agent = load_training_agent_checkpoint(
            Path(info.checkpoint_path),
            device=self.device,
            seed=self.seed + step,
        )
        freeze_agent(agent)
        info.loaded_agent = agent
        info.last_used = step

        return agent
    
    # ---------- PUBLIC INTERFACE ----------
    
    def sample_scripted_opponent(self, episode: int):
        """Sample from ``scripted`` only (ignores self-play / frozen)."""
        return self._create_scripted_opponent(episode)

    def sample_heuristic_opponent(self, episode: int):
        """Deprecated name for :meth:`sample_scripted_opponent`."""
        return self.sample_scripted_opponent(episode)
    
    def sample_opponent(self, episode: int):
        """
        Sample an opponent for the given episode.
        """
        config = self.self_play_config
        use_self_play = config is not None and episode >= (config.start_episode if config else 0)

        if use_self_play and config is not None:
            roll = random.random()
            threshold_current = config.current_self_fraction
            threshold_past = config.current_self_fraction + config.past_self_fraction

            if self.current_agent is not None and (roll < threshold_current or len(self.frozen_agents) == 0):
                return SelfPlayOpponent(self.current_agent, greedy=True)

            if roll < threshold_past and self.frozen_agents:
                info = self._sample_frozen_info()
                if info is not None:
                    return self._get_loaded_frozen_agent(info, step=episode)

        return self._create_scripted_opponent(episode)

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

    def get_frozen_stats_for_status(self) -> List[Dict[str, Any]]:
        """Per-frozen-agent stats (sofar) and current selection probability. Same weights as _sample_frozen_info."""
        if not self.frozen_agents:
            return []
        eps = 1e-2
        weights = [max(info.win_rate, eps) ** 2 for info in self.frozen_agents]
        total = sum(weights)
        probs = [w / total for w in weights] if total > 0 else [1.0 / len(self.frozen_agents)] * len(self.frozen_agents)
        out = []
        for info, p in zip(self.frozen_agents, probs):
            out.append({
                "checkpoint": os.path.basename(info.checkpoint_path),
                "episode": info.episode,
                "games": info.games,
                "wins": info.wins,
                "losses": info.losses,
                "draws": info.draws,
                "ema_win_rate": round(info.ema_win_rate, 4),
                "cumulative_win_rate": round(info.cumulative_win_rate, 4),
                "win_rate": round(info.ema_win_rate, 4),
                "selection_probability": round(p, 4),
            })
        return out
