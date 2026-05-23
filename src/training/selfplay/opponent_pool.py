"""Opponent pool for self-play training."""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from src.agents import RandomAgent, HeuristicAgent, SmartHeuristicAgent
from src.agents.othello import OthelloHeuristicAgent
from src.agents.base_agent import BaseAgent
from src.envs.base import StepResult
from src.utils import freeze_agent

from .league_policy import OpponentKind, decide_opponent_kind, pfsp_sample, sample_scripted_key, self_play_enabled
from .game_record import (
    GameRecord,
    SLOT_CURRENT,
    SLOT_SCRIPTED,
    build_scripted_slot_map,
    invert_scripted_slot_map,
    minibg_record_from_learner_score,
)
from .league_state import LeagueController, normalize_agent_score


@dataclass
class FrozenAgentInfo:
    """Frozen checkpoint entry; stats live in ``LeagueController``."""

    slot_id: int
    checkpoint_path: str
    episode: int
    loaded_agent: Optional[BaseAgent] = None
    last_used: int = 0
    _league: LeagueController = None  # type: ignore[assignment]

    def _slot(self):
        slot = self._league.get_slot(self.slot_id)
        if slot is None:
            raise RuntimeError(f"missing league slot {self.slot_id}")
        return slot

    @property
    def games(self) -> int:
        return self._slot().games

    @property
    def wins(self) -> int:
        return self._slot().wins

    @property
    def losses(self) -> int:
        return self._slot().losses

    @property
    def draws(self) -> int:
        return self._slot().draws

    @property
    def ema_win_rate(self) -> float:
        return self._slot().ema_win_rate

    @ema_win_rate.setter
    def ema_win_rate(self, value: float) -> None:
        self._league.set_ema_win_rate(self.slot_id, float(value))

    @property
    def cumulative_win_rate(self) -> float:
        return self._slot().cumulative_win_rate

    @property
    def win_rate(self) -> float:
        return self.ema_win_rate


@dataclass
class SelfPlayConfig:
    """Configuration options for self-play opponent sampling."""

    start_episode: int = 0
    current_self_fraction: float = 0.3
    past_self_fraction: float = 0.3
    max_frozen_agents: int = 10
    save_every: int = 1000
    frozen_ema_beta: float = 0.05
    rating: str = "ema"
    trueskill: Optional[Dict[str, float]] = None

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


ScriptedMode = Literal["classic", "minibg", "bglike"]


@dataclass(frozen=True)
class ScriptedOpponentsSpec:
    """Weights for sampling a non-learned opponent **inside the scripted branch**."""

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

    def save(self, path: str) -> None:  # pragma: no cover
        return None

    @classmethod
    def load(cls, path: str, **kwargs: Any) -> "SelfPlayOpponent":  # pragma: no cover
        raise NotImplementedError("SelfPlayOpponent cannot be loaded from disk.")

    def eval(self) -> None:
        return None

    def train(self) -> None:
        return None


class OpponentPool:
    """Pool of opponents for self-play training."""

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
        if scripted.mode in ("minibg", "bglike"):
            if scripted.mode == "minibg":
                from src.envs.minibg.heuristic_bots.bots import default_bot_constructors
            else:
                from src.envs.bglike.heuristic_bots import default_bot_constructors

            valid = frozenset(default_bot_constructors().keys())
            for name in scripted.distribution:
                if name != "random" and name not in valid:
                    raise ValueError(
                        f"{scripted.mode} scripted distribution: unknown key {name!r}; "
                        f"expected 'random' or {sorted(valid)}"
                    )

        self.self_play_config = self_play_config
        self.max_frozen_agents = (
            self_play_config.max_frozen_agents if self_play_config is not None else 0
        )
        self.current_agent = current_agent

        beta = float(self_play_config.frozen_ema_beta) if self_play_config else 0.05
        rating_kind = str(self_play_config.rating) if self_play_config else "ema"
        trueskill_cfg = dict(self_play_config.trueskill or {}) if self_play_config else None
        self._league = LeagueController(
            ema_beta=beta,
            rating_kind=rating_kind,
            trueskill=trueskill_cfg,
        )
        self._scripted_slot_ids = build_scripted_slot_map(self._scripted.distribution.keys())
        self._slot_id_to_scripted_key = invert_scripted_slot_map(self._scripted_slot_ids)
        self._league.register_scripted_slots(self._scripted_slot_ids)
        self._league.register_meta_slot(SLOT_CURRENT)

        self.frozen_agents: List[FrozenAgentInfo] = []
        default_scripted = next(iter(self._scripted_slot_ids.values()), SLOT_SCRIPTED)
        self._last_sample_slot_id: int = default_scripted
        self._last_scripted_key: Optional[str] = None
        self._pending_records: List[GameRecord] = []
        self._rng = random.Random(seed)

    def _create_scripted_opponent(self, episode: int, *, key: Optional[str] = None):
        bot_key = key or sample_scripted_key(self._scripted.distribution, self._rng)
        if self._scripted.mode == "minibg":
            from src.envs.minibg.heuristic_bots.agent_adapter import MiniBGHeuristicAgent
            from src.envs.minibg.heuristic_bots.tournament import make_bot

            rng_ep = self.seed + 100000 + episode
            if bot_key == "random":
                return RandomAgent(seed=self.seed + 200000 + episode)
            return MiniBGHeuristicAgent(make_bot(bot_key, rng_ep + 31))

        if self._scripted.mode == "bglike":
            from src.envs.bglike.heuristic_bots import make_heuristic_agent

            rng_ep = self.seed + 100000 + episode
            if bot_key == "random":
                return RandomAgent(seed=self.seed + 200000 + episode)
            return make_heuristic_agent(bot_key, seed=rng_ep + 31)

        opp_seed = self.seed + 100000 + episode
        if bot_key == "random":
            return RandomAgent(seed=opp_seed)
        if bot_key == "heuristic":
            return HeuristicAgent(seed=opp_seed)
        if bot_key == "smart_heuristic":
            return SmartHeuristicAgent(seed=opp_seed)
        if bot_key == "othello_heuristic":
            return OthelloHeuristicAgent(seed=opp_seed)
        raise ValueError(f"Unknown scripted opponent type: {bot_key}")

    def add_frozen_agent(self, checkpoint_path: str, episode: int) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        slot_id = self._league.add_frozen_checkpoint(checkpoint_path, episode)
        info = FrozenAgentInfo(
            slot_id=slot_id,
            checkpoint_path=checkpoint_path,
            episode=episode,
            _league=self._league,
        )
        self.frozen_agents.append(info)
        self._league.evict_worst(self.max_frozen_agents)
        self._sync_frozen_list()

    def _sync_frozen_list(self) -> None:
        live = {s.slot_id for s in self._league.frozen_slots()}
        self.frozen_agents = [fa for fa in self.frozen_agents if fa.slot_id in live]

    def submit(self, record: GameRecord) -> None:
        """Queue one game record for batch league update."""
        self._pending_records.append(record)

    def apply_outcomes(self, records: Sequence[GameRecord]) -> None:
        """Batch-update league stats (call after rollout, not per episode)."""
        self._league.apply_outcomes(records)

    def record_episode_outcome(self, agent_result: Optional[Union[int, float]]) -> None:
        """Record one game outcome for the last sampled opponent (pending until flush)."""
        self.record_outcome_for_slot(self._last_sample_slot_id, agent_result)

    def record_outcome_for_slot(
        self,
        slot_id: int,
        agent_result: Optional[Union[int, float]],
    ) -> None:
        """Record learner outcome vs a specific league slot (pending until flush)."""
        if agent_result is None:
            return
        score = normalize_agent_score(agent_result)
        sid = int(slot_id)
        self.submit(
            minibg_record_from_learner_score(
                sid,
                score,
                scripted_key=self._slot_id_to_scripted_key.get(sid),
            )
        )

    def flush_pending_outcomes(self) -> None:
        if self._pending_records:
            self.apply_outcomes(self._pending_records)
            self._pending_records.clear()

    def _sample_frozen_info(self) -> Optional[FrozenAgentInfo]:
        if not self.frozen_agents:
            return None
        slot_ids = [fa.slot_id for fa in self.frozen_agents]
        rates = [self._league.get_slot(sid).ema_win_rate for sid in slot_ids]  # type: ignore[union-attr]
        chosen = pfsp_sample(slot_ids, rates, self._rng)
        for info in self.frozen_agents:
            if info.slot_id == chosen:
                return info
        return self.frozen_agents[-1]

    def _get_loaded_frozen_agent(self, info: FrozenAgentInfo, step: int) -> BaseAgent:
        if info.loaded_agent is not None:
            info.last_used = step
            return info.loaded_agent

        loaded_infos = [fa for fa in self.frozen_agents if fa.loaded_agent is not None]
        if self.max_frozen_agents > 0 and len(loaded_infos) >= self.max_frozen_agents:
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

    def sample_scripted_opponent(self, episode: int):
        return self._create_scripted_opponent(episode)

    def sample_heuristic_opponent(self, episode: int):
        return self.sample_scripted_opponent(episode)

    def sample_opponent(self, episode: int):
        config = self.self_play_config
        use_self_play = self_play_enabled(
            episode=episode,
            start_episode=config.start_episode if config else 0,
            has_self_play_config=config is not None,
        )

        if use_self_play and config is not None:
            kind = decide_opponent_kind(
                self._rng.random(),
                current_fraction=config.current_self_fraction,
                past_fraction=config.past_self_fraction,
                frozen_nonempty=bool(self.frozen_agents),
                has_current_agent=self.current_agent is not None,
            )
            if kind == OpponentKind.CURRENT:
                self._last_sample_slot_id = SLOT_CURRENT
                return SelfPlayOpponent(self.current_agent, greedy=True)  # type: ignore[arg-type]
            if kind == OpponentKind.FROZEN:
                info = self._sample_frozen_info()
                if info is not None:
                    self._last_sample_slot_id = info.slot_id
                    return self._get_loaded_frozen_agent(info, step=episode)

        self._last_sample_slot_id = SLOT_SCRIPTED
        key = sample_scripted_key(self._scripted.distribution, self._rng)
        self._last_sample_slot_id = self._scripted_slot_ids[key]
        self._last_scripted_key = key
        return self._create_scripted_opponent(episode, key=key)

    def get_pool_stats(self) -> Dict[str, Any]:
        loaded_count = sum(1 for info in self.frozen_agents if info.loaded_agent is not None)
        return {
            "frozen_agents_count": len(self.frozen_agents),
            "loaded_agents_count": loaded_count,
            "self_play_enabled": self.self_play_config is not None,
            "self_play_ready": (
                self.self_play_config is not None and len(self.frozen_agents) > 0
            ),
        }

    def get_pool_stats_for_status(self) -> List[Dict[str, Any]]:
        return self._league.get_pool_stats_for_status()

    def get_status_file_data(self) -> Dict[str, Any]:
        return self._league.get_status_file_data()
