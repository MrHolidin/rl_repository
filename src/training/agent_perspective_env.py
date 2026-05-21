"""Single-agent gym-like wrapper over a two-player TurnBasedEnv.

Owns turn alternation, opponent sampling, randomized openings and reward
attribution from the agent's zero-sum perspective. The trainer becomes a
plain (s, a, r, s') loop on top of this wrapper.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from src.agents.base_agent import BaseAgent
from src.envs.minibg.structured_actions import StructAction
from src.envs.base import SingleAgentEnv, StepResult, TurnBasedEnv

BaseEnvType = TurnBasedEnv | SingleAgentEnv
from src.envs.reward_config import RewardConfig
from src.training.opponent_sampler import OpponentSampler
from src.training.random_opening import RandomOpeningConfig, maybe_apply_random_opening


# (info, agent_token) -> agent-perspective shaping for a non-terminal env step.
ShapingFn = Callable[[Dict[str, Any], int], float]


def make_minibg_shaping_fn(scale: float) -> Optional[ShapingFn]:
    """Return a shaping fn that reads ``info["battle_signed"]`` from the agent side.

    ``info["battle_signed"]`` is ``(signed_p0, signed_p1)`` where ``signed_pi``
    is positive iff player ``i`` dealt more damage than they received. The
    agent token +1 maps to player 0, -1 to player 1.
    """
    if not scale:
        return None

    def _fn(info: Dict[str, Any], agent_token: int) -> float:
        bs = info.get("battle_signed")
        if bs is None:
            return 0.0
        idx = 0 if agent_token == 1 else 1
        return float(scale) * float(bs[idx])

    return _fn


class AgentPerspectiveEnv(SingleAgentEnv):
    """Wraps a 2-player ``TurnBasedEnv`` plus an ``OpponentSampler`` into a
    single-agent env.

    ``step(action)`` advances the underlying env until the next agent decision
    point or terminal, accumulating reward in the agent's zero-sum perspective.
    ``reset()`` samples a new opponent and side, applies any random opening,
    and drains opponent moves until the agent must act (or the episode ended
    inside the opening sequence).
    """

    MAX_OPPONENT_DRAIN_STEPS = 10_000
    MAX_RESET_RETRIES = 64
    _SEED_SPACE = 2**31 - 1

    def __init__(
        self,
        base_env: BaseEnvType,
        opponent_sampler: OpponentSampler,
        *,
        agent_first_probability: float = 0.5,
        rng: Optional[random.Random] = None,
        random_opening_config: Optional[RandomOpeningConfig] = None,
        shaping_fn: Optional[ShapingFn] = None,
        reward_config: Optional[RewardConfig] = None,
    ) -> None:
        self.base = base_env
        self.opponent_sampler = opponent_sampler
        self._agent_first_probability = float(agent_first_probability)
        self._rng = rng if rng is not None else random.Random()
        self.random_opening_config = random_opening_config
        self.shaping_fn = shaping_fn
        self.reward_config = (
            reward_config
            if reward_config is not None
            else getattr(base_env, "reward_config", None)
        )
        self._opponent: Optional[BaseAgent] = None
        self._agent_token: int = 1
        self._episode_index: int = 0
        self._done: bool = False

    # ------------------------------------------------------------------
    # SingleAgentEnv interface
    # ------------------------------------------------------------------

    @property
    def legal_actions_mask(self) -> np.ndarray:
        return self.base.legal_actions_mask

    @property
    def state(self) -> Any:
        """Underlying turn env state (e.g. ``MiniBGState`` for MiniBG)."""
        return self.base.state

    def legal_structured_actions(self):
        """Delegate to base when implemented (``MiniBGEnv``)."""
        if not hasattr(self.base, "legal_structured_actions"):
            raise AttributeError(
                f"{type(self.base).__name__} has no legal_structured_actions; "
                "structured MiniBG PPO requires MiniBGEnv as base."
            )
        return self.base.legal_structured_actions()

    @property
    def done(self) -> bool:
        return self._done

    @property
    def agent_token(self) -> int:
        return self._agent_token

    def reset(self) -> np.ndarray:
        # Episodes that terminate before the agent ever acts (e.g. random
        # opening rolls into a battle) are silently retried; they carry no
        # learning signal for the agent. Bounded to avoid pathological loops.
        for _ in range(self.MAX_RESET_RETRIES):
            # Per-episode seed: deterministic given `self._rng` (which the
            # caller seeds from app config), but distinct across episodes so
            # the underlying game's stored constructor seed cannot pin every
            # reset to the same shop / initiative / battle RNG state.
            episode_seed = self._rng.randrange(self._SEED_SPACE)
            obs = self.base.reset(seed=episode_seed)

            if self.random_opening_config is not None:
                obs, _moves, prologue_done = maybe_apply_random_opening(
                    env=self.base,
                    initial_obs=obs,
                    config=self.random_opening_config,
                    rng=self._rng,
                )
                if prologue_done:
                    continue

            self._agent_token = self._roll_agent_token()
            self.opponent_sampler.prepare(self._episode_index)
            self._opponent = self.opponent_sampler.sample()
            if hasattr(self._opponent, "set_env"):
                self._opponent.set_env(self.base)
            if hasattr(self._opponent, "epsilon"):
                setattr(self._opponent, "epsilon", 0.0)

            obs, _r, drain_done, _ = self._drain_opponent(obs)
            if drain_done:
                # Opponent ended the episode on its own; agent had no decision
                # point. Retry with a fresh episode.
                self._episode_index += 1
                continue

            self._done = False
            return obs
        raise RuntimeError(
            "AgentPerspectiveEnv: could not obtain a non-terminal initial state."
        )

    def step(self, action: int) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done; call reset() first.")

        step = self.base.step(action)
        reward = self._reward_in_agent_perspective(step, agent_acted=True)
        if step.done:
            self._done = True
            return StepResult(step.obs, reward, True, False, step.info)

        obs, opp_reward, drain_done, drain_info = self._drain_opponent(step.obs)
        if drain_done:
            self._done = True
            if drain_info is None:
                raise RuntimeError(
                    "AgentPerspectiveEnv.step: episode ended in opponent drain "
                    "but terminal info is missing"
                )
            out_info = drain_info
        else:
            out_info = step.info
        return StepResult(
            obs=obs,
            reward=reward + opp_reward,
            terminated=drain_done,
            truncated=False,
            info=out_info,
        )

    def step_structured(
        self,
        action: StructAction,
        *,
        board_perm: Optional[Tuple[int, ...]] = None,
    ) -> StepResult:
        """Like ``step`` but uses ``MiniBGEnv.step_structured`` on the base env."""
        if self._done:
            raise RuntimeError("Episode is done; call reset() first.")
        if not hasattr(self.base, "step_structured"):
            raise AttributeError(
                f"{type(self.base).__name__} has no step_structured; structured MiniBG PPO requires MiniBGEnv."
            )

        step = self.base.step_structured(action, board_perm=board_perm)
        reward = self._reward_in_agent_perspective(step, agent_acted=True)
        if step.done:
            self._done = True
            return StepResult(step.obs, reward, True, False, step.info)

        obs, opp_reward, drain_done, drain_info = self._drain_opponent(step.obs)
        if drain_done:
            self._done = True
            if drain_info is None:
                raise RuntimeError(
                    "AgentPerspectiveEnv.step_structured: episode ended in opponent drain "
                    "but terminal info is missing"
                )
            out_info = drain_info
        else:
            out_info = step.info
        return StepResult(
            obs=obs,
            reward=reward + opp_reward,
            terminated=drain_done,
            truncated=False,
            info=out_info,
        )

    def notify_episode_end(self, info: Dict[str, Any]) -> None:
        if self.opponent_sampler is None:
            self._episode_index += 1
            return
        agent_result = self._agent_relative_result(info.get("winner"))
        episode_info = {
            "agent_token": self._agent_token,
            "agent_result": agent_result,
            "info": info,
        }
        self.opponent_sampler.on_episode_end(self._episode_index, episode_info)
        self._episode_index += 1

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _delegates_opponent_play(self) -> bool:
        return bool(getattr(self.base, "delegates_opponent_play", False))

    def _drain_opponent(
        self, obs: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Optional[Dict[str, Any]]]:
        """Drain opponent plies until agent's turn or episode end.

        If the episode ends on an opponent ply, returns that ply's ``info`` as the
        fourth element so ``step`` / ``step_structured`` do not surface a stale
        pre-drain ``info`` without ``winner``.
        """
        if self._delegates_opponent_play():
            if self.base.done:
                last = getattr(self.base, "_last_info", {})
                return obs, 0.0, True, last if isinstance(last, dict) else {}
            return obs, 0.0, False, None

        accumulated = 0.0
        steps = 0
        while (
            not self.base.done
            and not self._is_agent_turn()
            and steps < self.MAX_OPPONENT_DRAIN_STEPS
        ):
            steps += 1
            opp_mask = self.base.legal_actions_mask
            if not bool(opp_mask.any()):
                from src.envs.minibg.invariants import assert_shop_has_legal_actions

                assert_shop_has_legal_actions(
                    self.base.state,
                    [],
                    where=f"AgentPerspectiveEnv._drain_opponent step={steps}",
                )
            if hasattr(self._opponent, "opponent_step"):
                opp_step = self._opponent.opponent_step(
                    self.base,
                    obs,
                    legal_mask=opp_mask,
                    deterministic=False,
                )
            else:
                opp_action = self._opponent.act(obs, legal_mask=opp_mask, deterministic=False)
                from src.envs.minibg.invariants import assert_action_in_legal_mask

                assert_action_in_legal_mask(
                    self.base.state,
                    int(opp_action),
                    opp_mask,
                    where=f"AgentPerspectiveEnv._drain_opponent step={steps}",
                    rl_pending=getattr(self.base, "_rl_pending", None) is not None,
                )
                opp_step = self.base.step(opp_action)
            obs = opp_step.obs
            accumulated += self._reward_in_agent_perspective(opp_step, agent_acted=False)
            if opp_step.done:
                term = opp_step.info if isinstance(opp_step.info, dict) else {}
                return obs, accumulated, True, term
        return obs, accumulated, False, None

    def _reward_in_agent_perspective(self, step: StepResult, agent_acted: bool) -> float:
        if step.done:
            return self._final_reward_for_agent(step.info)
        if self.shaping_fn is None:
            return 0.0
        return float(self.shaping_fn(step.info, self._agent_token))

    def _final_reward_for_agent(self, info: Dict[str, Any]) -> float:
        if isinstance(info, dict) and info.get("placement_reward") is not None:
            return float(info["placement_reward"])
        if isinstance(info, dict) and info.get("placements_current"):
            placements = info["placements_current"]
            if placements:
                from src.envs.bglike.placement import placement_reward

                return float(
                    sum(placement_reward(int(p)) for p in placements.values())
                    / len(placements)
                )
        winner = info.get("winner") if isinstance(info, dict) else None
        rc = self.reward_config
        if rc is not None:
            if winner == 0 or winner is None:
                return float(getattr(rc, "draw", 0.0))
            if winner == self._agent_token:
                return float(getattr(rc, "win", 1.0))
            return float(getattr(rc, "loss", -1.0))
        if winner == 0 or winner is None:
            return 0.0
        if winner == self._agent_token:
            return 1.0
        return -1.0

    def _agent_relative_result(self, winner) -> Optional[int]:
        if winner is None:
            return None
        if winner == 0:
            return 0
        if winner == self._agent_token:
            return 1
        if winner == -self._agent_token:
            return -1
        return None

    def _is_agent_turn(self) -> bool:
        if self._delegates_opponent_play():
            acting = getattr(self.base, "acting_seat", None)
            return not self.base.done and acting is not None
        token = getattr(self.base, "current_player_token", None)
        if token is None:
            return self._agent_token == 1
        return int(token) == self._agent_token

    def _roll_agent_token(self) -> int:
        p = self._agent_first_probability
        if p >= 1.0:
            return 1
        if p <= 0.0:
            return -1
        return 1 if self._rng.random() < p else -1


__all__ = [
    "AgentPerspectiveEnv",
    "BaseEnvType",
    "ShapingFn",
    "make_minibg_shaping_fn",
]
