from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..base import StepResult, TurnBasedEnv
from ..reward_config import RewardConfig
from .action_map import (
    A_BUY_BASE,
    A_DISCOVER_BASE,
    A_FINISH,
    A_FINISH_FREEZE_SHOP,
    A_LEVEL_UP,
    A_MAGNET_BASE,
    A_PLACE_BASE,
    A_ROLL,
    A_SELL_BASE,
    A_SWAP_BOARD_0,
    NUM_ENV_ACTIONS,
    NUM_SWAP_ADJ,
    env_action_to_game_action,
    is_swap_board,
    swap_adj_index_from_env_action,
)
from .actions import (
    BOARD_SIZE,
    DAMAGE_CAP,
    HAND_SIZE,
    MAX_SHOP_SLOTS,
    Action as GameAction,
    magnet_game_action,
)
from .game import MiniBGGame, PLAYER_TOKENS
from .obs import OBS_DIM, build_observation
from .state import MiniBGState, Minion, PlayerPhase, Race
from .structured_actions import (
    StructAction,
    StructActionType,
    structured_action_to_replay_env_int,
    structured_legal_set,
    validate_board_perm,
)


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
        shop_excluded_race: Optional[Race] = None,
        shop_full_tribes: bool = False,
    ) -> None:
        self._seed = seed
        # Stored for backward compatibility / introspection only.
        self._battle_damage_shaping = float(battle_damage_shaping)
        self.reward_config = reward_config or RewardConfig(
            invalid_action=INVALID_ACTION_REWARD
        )
        self._shop_excluded_race = shop_excluded_race
        self._shop_full_tribes = shop_full_tribes
        self._game: MiniBGGame = MiniBGGame(
            seed=seed,
            shop_excluded_race=shop_excluded_race,
            shop_full_tribes=shop_full_tribes,
        )
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
            self._game = MiniBGGame(
                seed=seed,
                shop_excluded_race=self._shop_excluded_race,
                shop_full_tribes=self._shop_full_tribes,
            )
        elif self._seed is not None:
            self._game = MiniBGGame(
                seed=self._seed,
                shop_excluded_race=self._shop_excluded_race,
                shop_full_tribes=self._shop_full_tribes,
            )
        else:
            self._game = MiniBGGame(
                shop_excluded_race=self._shop_excluded_race,
                shop_full_tribes=self._shop_full_tribes,
            )
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

        if is_swap_board(action_int):
            i = swap_adj_index_from_env_action(action_int)
            self._state = self._game.swap_board_adjacent(self._state, acting_idx, i)
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

    def legal_structured_actions(self) -> List[StructAction]:
        """Legal structured tokens for the current player (shop actions + COMPLETE_TURN)."""
        if self._state is None or self._state.done:
            return []

        player = self._state.players[self._state.current_player_index]
        legal_game = set(int(a) for a in self._game.legal_actions(self._state))
        out: List[StructAction] = []

        if player.pending_choice is not None:
            for i in range(3):
                if int(GameAction.DISCOVER_PICK_0) + i in legal_game:
                    out.append(StructAction(StructActionType.DISCOVER_PICK, (i,)))
            return out

        if int(GameAction.ROLL) in legal_game:
            out.append(StructAction(StructActionType.ROLL))
        if int(GameAction.LEVEL_UP) in legal_game:
            out.append(StructAction(StructActionType.LEVEL_UP))

        for slot in range(MAX_SHOP_SLOTS):
            if (int(GameAction.BUY_SLOT_0) + slot) in legal_game:
                out.append(StructAction(StructActionType.BUY, (slot,)))

        for pos in range(BOARD_SIZE):
            if (int(GameAction.SELL_BOARD_0) + pos) in legal_game:
                out.append(StructAction(StructActionType.SELL, (pos,)))

        for h in range(HAND_SIZE):
            if (int(GameAction.PLACE_HAND_0) + h) in legal_game:
                out.append(StructAction(StructActionType.PLACE, (h,)))

        for h in range(HAND_SIZE):
            for b in range(BOARD_SIZE):
                mg = int(magnet_game_action(h, b))
                if mg in legal_game:
                    out.append(StructAction(StructActionType.MAGNET, (h, b)))

        if int(GameAction.FINISH) in legal_game:
            out.append(StructAction(StructActionType.COMPLETE_TURN))
        if int(GameAction.FINISH_FREEZE_SHOP) in legal_game:
            out.append(StructAction(StructActionType.COMPLETE_TURN_FREEZE_SHOP))

        return out

    def step_structured(
        self,
        action: StructAction,
        *,
        board_perm: Optional[Tuple[int, ...]] = None,
    ) -> StepResult:
        """Apply a structured action. COMPLETE_TURN requires ``board_perm`` (length ``BOARD_SIZE`` permutation)."""
        if self._state is None:
            raise ValueError("Environment not initialized; call reset() first.")
        if self._state.done:
            raise ValueError("Episode is done. Call reset() first.")

        acting_idx_before = self._state.current_player_index
        legal = structured_legal_set(tuple(self.legal_structured_actions()))

        def _illegal_reply() -> StepResult:
            info_il: Dict[str, Any] = {
                "winner": None,
                "termination_reason": "illegal",
                "invalid_action": True,
            }
            if self._replay_sink is not None:
                assert self._state is not None
                self._replay_frame += 1
                rep_a = structured_action_to_replay_env_int(action)
                self._replay_sink.frame(
                    episode=self._replay_episode,
                    frame=self._replay_frame,
                    acting_idx=acting_idx_before,
                    action=rep_a,
                    illegal=True,
                    state=self._state,
                    info=info_il,
                )
            return StepResult(
                obs=self._get_obs(),
                reward=float(self.reward_config.invalid_action),
                terminated=False,
                truncated=False,
                info=info_il,
            )

        if action not in legal:
            return _illegal_reply()

        if action.type in (
            StructActionType.COMPLETE_TURN,
            StructActionType.COMPLETE_TURN_FREEZE_SHOP,
        ):
            if board_perm is None:
                return _illegal_reply()
            try:
                validate_board_perm(tuple(board_perm))
            except ValueError:
                return _illegal_reply()
        elif board_perm is not None:
            return _illegal_reply()

        acting_idx = self._state.current_player_index
        acting_token = PLAYER_TOKENS[acting_idx]
        prev_round = self._state.round_number
        prev_done = self._state.done
        prev_hp = (
            self._state.players[0].health,
            self._state.players[1].health,
        )

        try:
            if action.type in (
                StructActionType.COMPLETE_TURN,
                StructActionType.COMPLETE_TURN_FREEZE_SHOP,
            ):
                assert board_perm is not None
                perm_tuple = tuple(int(x) for x in board_perm)
                shop_finish = (
                    int(GameAction.FINISH_FREEZE_SHOP)
                    if action.type == StructActionType.COMPLETE_TURN_FREEZE_SHOP
                    else int(GameAction.FINISH)
                )
                self._state = self._apply_complete_turn(
                    self._state,
                    acting_idx,
                    perm_tuple,
                    shop_phase_finish=int(shop_finish),
                )
            else:
                ga = self._struct_action_to_game_action(action)
                self._state = self._game.apply_action(self._state, ga)
        except ValueError:
            return _illegal_reply()

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

        replay_action = structured_action_to_replay_env_int(action)
        if self._replay_sink is not None:
            assert self._state is not None
            self._replay_frame += 1
            self._replay_sink.frame(
                episode=self._replay_episode,
                frame=self._replay_frame,
                acting_idx=acting_idx,
                action=replay_action,
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

    def _apply_complete_turn(
        self,
        state: MiniBGState,
        acting_idx: int,
        perm: Tuple[int, ...],
        *,
        shop_phase_finish: int,
    ) -> MiniBGState:
        """Optional ``reorder_board`` then shop exit (FINISH or FINISH_FREEZE_SHOP)."""
        player = state.players[acting_idx]
        if player.phase != PlayerPhase.SHOP:
            raise ValueError(
                f"COMPLETE_TURN invalid phase {player.phase.name}"
            )
        new_state = state
        k = len(player.board)
        identity = tuple(range(BOARD_SIZE))
        if perm != identity and k > 0:
            new_state = self._game.reorder_board(new_state, acting_idx, perm)
        return self._game.apply_action(new_state, shop_phase_finish)

    @staticmethod
    def _struct_action_to_game_action(action: StructAction) -> int:
        if action.type == StructActionType.ROLL:
            return int(GameAction.ROLL)
        if action.type == StructActionType.LEVEL_UP:
            return int(GameAction.LEVEL_UP)
        if action.type == StructActionType.BUY:
            return int(GameAction.BUY_SLOT_0) + action.args[0]
        if action.type == StructActionType.SELL:
            return int(GameAction.SELL_BOARD_0) + action.args[0]
        if action.type == StructActionType.PLACE:
            return int(GameAction.PLACE_HAND_0) + action.args[0]
        if action.type == StructActionType.MAGNET:
            return int(magnet_game_action(action.args[0], action.args[1]))
        if action.type == StructActionType.DISCOVER_PICK:
            return int(GameAction.DISCOVER_PICK_0) + action.args[0]
        raise ValueError(f"not a shop-phase structured action: {action}")

    @property
    def legal_actions_mask(self) -> np.ndarray:
        if self._state is None:
            return np.zeros(NUM_ENV_ACTIONS, dtype=bool)
        mask = np.zeros(NUM_ENV_ACTIONS, dtype=bool)
        if self._state.done:
            return mask

        player = self._state.players[self._state.current_player_index]

        if player.phase == PlayerPhase.SHOP:
            k = len(player.board)
            for i in range(NUM_SWAP_ADJ):
                if i + 1 < k:
                    mask[A_SWAP_BOARD_0 + i] = True

        # Shop phase: bridge from game's legal_actions.
        legal_game = set(int(a) for a in self._game.legal_actions(self._state))

        if int(GameAction.ROLL) in legal_game:
            mask[A_ROLL] = True
        if int(GameAction.LEVEL_UP) in legal_game:
            mask[A_LEVEL_UP] = True

        for slot in range(MAX_SHOP_SLOTS):
            if (int(GameAction.BUY_SLOT_0) + slot) in legal_game:
                mask[A_BUY_BASE + slot] = True

        for pos in range(BOARD_SIZE):
            if (int(GameAction.SELL_BOARD_0) + pos) in legal_game:
                mask[A_SELL_BASE + pos] = True

        for h in range(HAND_SIZE):
            if (int(GameAction.PLACE_HAND_0) + h) in legal_game:
                mask[A_PLACE_BASE + h] = True

        for h in range(HAND_SIZE):
            for b in range(BOARD_SIZE):
                mg = int(magnet_game_action(h, b))
                if mg in legal_game:
                    mask[A_MAGNET_BASE + h * BOARD_SIZE + b] = True

        for i in range(3):
            if int(GameAction.DISCOVER_PICK_0) + i in legal_game:
                mask[A_DISCOVER_BASE + i] = True

        if int(GameAction.FINISH) in legal_game:
            mask[A_FINISH] = True
        if int(GameAction.FINISH_FREEZE_SHOP) in legal_game:
            mask[A_FINISH_FREEZE_SHOP] = True

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
