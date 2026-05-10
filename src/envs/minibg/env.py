from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from ..base import StepResult, TurnBasedEnv
from ..reward_config import RewardConfig
from .action_map import (
    A_BUY_BASE,
    A_FINISH,
    A_LEVEL_UP,
    A_PLACE_BASE,
    A_ROLL,
    A_SELECT_ORDER_BASE,
    A_SELL_BASE,
    NUM_ENV_ACTIONS,
    PERMUTATIONS_4,
    env_action_to_game_action,
    is_select_order,
    legal_order_indices,
    order_index,
)
from .actions import (
    BOARD_SIZE,
    DAMAGE_CAP,
    HAND_SIZE,
    SHOP_SIZE,
    Action as GameAction,
)
from .game import MiniBGGame, PLAYER_TOKENS
from .obs import OBS_DIM, build_observation
from .state import MiniBGState, Minion, PlayerPhase


INVALID_ACTION_REWARD = -1.0


class MiniBGEnv(TurnBasedEnv):
    """RL wrapper around MiniBGGame.

    Observation: self-centric, fixed-length vector (see obs.py).
    Terminal reward: +1 / -1 / 0 from the acting player's perspective.
    Illegal actions use ``reward_config.invalid_action``.
    On each battle, ``info["battle_signed"] = (signed_p0, signed_p1)`` is emitted
    so symmetric per-player shaping can be applied externally (see
    ``src.training.agent_perspective_env``). The constructor still accepts
    ``battle_damage_shaping`` for backward compatibility but does NOT add shaping
    to ``step().reward`` -- shaping is the wrapper's responsibility.
    Optional ``replay_path`` / ``replay_meta``: append JSONL frames per step (see ``replay.py``).
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        *,
        battle_damage_shaping: float = 0.0,
        reward_config: Optional[RewardConfig] = None,
        replay_path: Optional[Union[str, Path]] = None,
        replay_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._seed = seed
        # Stored for backward compatibility / introspection only.
        self._battle_damage_shaping = float(battle_damage_shaping)
        self.reward_config = reward_config or RewardConfig(
            invalid_action=INVALID_ACTION_REWARD
        )
        self._game: MiniBGGame = MiniBGGame(seed=seed)
        self._state: Optional[MiniBGState] = None
        self._last_seen_enemy_board: List[List[Minion]] = [[], []]
        self._last_battle_signed: List[float] = [0.0, 0.0]
        self._replay_sink: Any = None
        self._replay_episode = -1
        self._replay_frame = 0
        if replay_path is not None:
            from .replay import ReplayJsonlSink

            self._replay_sink = ReplayJsonlSink(
                replay_path, {"format": 2, "game": "minibg", **(replay_meta or {})}
            )
        self.reset(seed=seed)

    # ------------------------------------------------------------------
    # TurnBasedEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._seed = seed
            self._game = MiniBGGame(seed=seed)
        elif self._seed is not None:
            self._game = MiniBGGame(seed=self._seed)
        self._state = self._game.initial_state()
        self._last_seen_enemy_board = [[], []]
        self._last_battle_signed = [0.0, 0.0]
        if self._replay_sink is not None:
            self._replay_episode += 1
            self._replay_sink.episode_break(self._replay_episode)
            self._replay_frame = 0
        return self._get_obs()

    def close_replay(self) -> None:
        if self._replay_sink is not None:
            self._replay_sink.close()
            self._replay_sink = None

    def step(self, action: int) -> StepResult:
        if self._state is None:
            raise ValueError("Environment not initialized; call reset() first.")
        if self._state.done:
            raise ValueError("Episode is done. Call reset() first.")

        action_int = int(action)
        acting_idx_before = self._state.current_player_index
        legal_mask = self.legal_actions_mask
        if (
            action_int < 0
            or action_int >= NUM_ENV_ACTIONS
            or not bool(legal_mask[action_int])
        ):
            info: Dict[str, Any] = {
                "winner": None,
                "termination_reason": "illegal",
                "invalid_action": True,
            }
            if self._replay_sink is not None:
                assert self._state is not None
                self._replay_frame += 1
                self._replay_sink.frame(
                    episode=self._replay_episode,
                    frame=self._replay_frame,
                    acting_idx=acting_idx_before,
                    action=action_int,
                    illegal=True,
                    state=self._state,
                    info=info,
                )
            return StepResult(
                obs=self._get_obs(),
                reward=float(self.reward_config.invalid_action),
                terminated=False,
                truncated=False,
                info=info,
            )

        acting_idx = self._state.current_player_index
        acting_token = PLAYER_TOKENS[acting_idx]
        prev_round = self._state.round_number
        prev_done = self._state.done
        prev_hp = (
            self._state.players[0].health,
            self._state.players[1].health,
        )

        if is_select_order(action_int):
            j = order_index(action_int)
            perm = PERMUTATIONS_4[j]
            self._state = self._game.reorder_board(self._state, acting_idx, perm)
            self._state = self._game.apply_action(
                self._state, int(GameAction.FINISH)
            )
        else:
            game_action = env_action_to_game_action(action_int)
            self._state = self._game.apply_action(self._state, game_action)

        battle_happened = (
            self._state.round_number != prev_round
            or (self._state.done and not prev_done)
        )
        if battle_happened:
            self._update_battle_summary(prev_hp)

        reward = 0.0
        terminated = False
        info: Dict[str, Any] = {
            "winner": self._state.winner,
            "termination_reason": None,
            "invalid_action": False,
            # `battle_damage_shaping` kept as a 0.0 placeholder for replay schema
            # backward compatibility; actual shaping is computed by the trainer wrapper.
            "battle_damage_shaping": 0.0,
        }
        if battle_happened:
            info["battle_signed"] = (
                float(self._last_battle_signed[0]),
                float(self._last_battle_signed[1]),
            )
        if self._state.done:
            terminated = True
            if self._state.winner == acting_token:
                reward = 1.0
                info["termination_reason"] = "win"
            elif self._state.winner == -acting_token:
                reward = -1.0
                info["termination_reason"] = "loss"
            else:
                reward = 0.0
                info["termination_reason"] = "draw"

        if self._replay_sink is not None:
            assert self._state is not None
            self._replay_frame += 1
            self._replay_sink.frame(
                episode=self._replay_episode,
                frame=self._replay_frame,
                acting_idx=acting_idx,
                action=action_int,
                illegal=False,
                state=self._state,
                info=info,
            )

        return StepResult(
            obs=self._get_obs(),
            reward=reward,
            terminated=terminated,
            truncated=False,
            info=info,
        )

    @property
    def legal_actions_mask(self) -> np.ndarray:
        if self._state is None:
            return np.zeros(NUM_ENV_ACTIONS, dtype=bool)
        mask = np.zeros(NUM_ENV_ACTIONS, dtype=bool)
        if self._state.done:
            return mask

        player = self._state.players[self._state.current_player_index]

        # Order phase: SELECT_ORDER_* are the only legal env actions. Only
        # ``k!`` canonical perms are legal (one representative per equivalence
        # class under compact-after-permute) — exposing the redundant 24
        # outputs to DQN washes out the gradient signal in this phase.
        if player.phase == PlayerPhase.ORDER:
            k = len(player.board)
            for j in legal_order_indices(k):
                mask[A_SELECT_ORDER_BASE + j] = True
            return mask

        # Shop phase: bridge from game's legal_actions.
        legal_game = set(int(a) for a in self._game.legal_actions(self._state))

        if int(GameAction.ROLL) in legal_game:
            mask[A_ROLL] = True
        if int(GameAction.LEVEL_UP) in legal_game:
            mask[A_LEVEL_UP] = True

        for slot in range(SHOP_SIZE):
            if (int(GameAction.BUY_SLOT_0) + slot) in legal_game:
                mask[A_BUY_BASE + slot] = True

        for pos in range(BOARD_SIZE):
            if (int(GameAction.SELL_BOARD_0) + pos) in legal_game:
                mask[A_SELL_BASE + pos] = True

        for h in range(HAND_SIZE):
            if (int(GameAction.PLACE_HAND_0) + h) in legal_game:
                mask[A_PLACE_BASE + h] = True

        if int(GameAction.FINISH) in legal_game:
            mask[A_FINISH] = True

        return mask

    def current_player(self) -> int:
        assert self._state is not None
        return self._state.current_player_index

    @property
    def current_player_token(self) -> int:
        return PLAYER_TOKENS[self.current_player()]

    @property
    def winner(self) -> Optional[int]:
        return None if self._state is None else self._state.winner

    @property
    def done(self) -> bool:
        return False if self._state is None else self._state.done

    @property
    def state(self) -> MiniBGState:
        if self._state is None:
            raise RuntimeError("Environment not initialized; call reset() first.")
        return self._state

    @property
    def game(self) -> MiniBGGame:
        return self._game

    def get_legal_actions(self) -> Sequence[int]:
        """Integer env actions (for ``maybe_apply_random_opening`` / generic eval)."""
        return [int(i) for i in np.flatnonzero(self.legal_actions_mask)]

    def render(self, mode: str = "human") -> Optional[str]:
        if mode != "human" or self._state is None:
            return None
        s = self._state
        for i, p in enumerate(s.players):
            print(
                f"P{i} hp={p.health} gold={p.gold} tier={p.tavern_tier} "
                f"finished={p.shopping_finished} acts_used={p.shop_actions_used}"
            )
            print(f"  board: {[m.card_id for m in p.board]}")
            print(
                f"  shop:  "
                f"{[None if m is None else m.card_id for m in p.shop]}"
            )
        print(
            f"round={s.round_number} cur_idx={s.current_player_index} "
            f"init_p={s.initiative_player} done={s.done} winner={s.winner}"
        )
        return None

    def get_state_hash(self) -> str:
        assert self._state is not None
        s = self._state
        parts = [
            str(s.round_number),
            str(s.current_player_index),
            str(s.initiative_player),
            str(int(s.done)),
            str(s.winner),
        ]
        for p in s.players:
            parts.append(
                f"{p.health}|{p.gold}|{p.tavern_tier}|"
                f"{int(p.shopping_finished)}|{p.shop_actions_used}"
            )
            parts.append(",".join(self._minion_repr(m) for m in p.board))
            parts.append(
                ",".join(
                    "_" if m is None else self._minion_repr(m) for m in p.shop
                )
            )
        return ";".join(parts)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _update_battle_summary(self, prev_hp: tuple) -> None:
        assert self._state is not None
        new_hp = (
            self._state.players[0].health,
            self._state.players[1].health,
        )
        dmg0 = max(0, prev_hp[0] - new_hp[0])
        dmg1 = max(0, prev_hp[1] - new_hp[1])
        self._last_battle_signed[0] = (dmg1 - dmg0) / DAMAGE_CAP
        self._last_battle_signed[1] = (dmg0 - dmg1) / DAMAGE_CAP
        self._last_seen_enemy_board[0] = [
            copy(m) for m in self._state.players[1].board
        ]
        self._last_seen_enemy_board[1] = [
            copy(m) for m in self._state.players[0].board
        ]

    def _get_obs(self) -> np.ndarray:
        assert self._state is not None
        idx = self._state.current_player_index
        return build_observation(
            self._state,
            idx,
            self._last_battle_signed[idx],
            self._last_seen_enemy_board[idx],
        )

    @staticmethod
    def _minion_repr(m: Minion) -> str:
        s = "S" if m.has_shield else ""
        return f"{m.card_id}:{m.bonus_attack}+{m.bonus_health}{s}"


__all__ = ["MiniBGEnv", "INVALID_ACTION_REWARD"]
