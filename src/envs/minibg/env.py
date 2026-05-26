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
    is_apply_effect_skip,
    is_place,
    is_target_board,
    is_swap_board,
    place_slot,
    swap_adj_index_from_env_action,
    target_board_slot,
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
from .state import MiniBGState, Minion, PendingChoiceKind, PlayerPhase, Race
from .rl_place import (
    RlPlacePlan,
    commit_rl_place_plan,
    commit_simple_place_from_hand,
    open_rl_place_plan,
)
from .structured_actions import (
    StructAction,
    StructActionType,
    structured_action_to_replay_env_int,
    structured_legal_set,
    validate_board_perm,
)
from src.bg_recruitment.triples import flush_triple_reward_queue_if_idle


INVALID_ACTION_REWARD = -1.0


class MiniBGEnv(TurnBasedEnv):
    """RL wrapper around MiniBGGame.

    Observation: self-centric, fixed-length vector (see obs.py).
    Terminal reward: +1 / -1 / 0 from the acting player's perspective.
    Illegal actions raise ``RuntimeError`` (see ``invariants.raise_illegal_env_action``).
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
        shop_excluded_count: Optional[int] = None,
        shop_full_tribes: bool = False,
        patch_dir: Optional[str] = None,
    ) -> None:
        self._seed = seed
        # Stored for backward compatibility / introspection only.
        self._battle_damage_shaping = float(battle_damage_shaping)
        self.reward_config = reward_config or RewardConfig(
            invalid_action=INVALID_ACTION_REWARD
        )
        self._shop_excluded_race = shop_excluded_race
        self._shop_excluded_count = shop_excluded_count
        self._shop_full_tribes = shop_full_tribes
        self._patch_dir = patch_dir
        self._game: MiniBGGame = MiniBGGame(
            seed=seed,
            shop_excluded_race=shop_excluded_race,
            shop_excluded_count=shop_excluded_count,
            shop_full_tribes=shop_full_tribes,
            patch_dir=patch_dir,
        )
        self._state: Optional[MiniBGState] = None
        self._last_seen_enemy_board: List[List[Minion]] = [[], []]
        self._last_battle_signed: List[float] = [0.0, 0.0]
        self._replay_sink: Any = None
        self._replay_episode = -1
        self._replay_frame = 0
        self._rl_pending: Optional[RlPlacePlan] = None
        self._rl_place_budget_pending = False
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
                shop_excluded_count=self._shop_excluded_count,
                shop_full_tribes=self._shop_full_tribes,
                patch_dir=self._patch_dir,
            )
        elif self._seed is not None:
            self._game = MiniBGGame(
                seed=self._seed,
                shop_excluded_race=self._shop_excluded_race,
                shop_excluded_count=self._shop_excluded_count,
                shop_full_tribes=self._shop_full_tribes,
                patch_dir=self._patch_dir,
            )
        else:
            self._game = MiniBGGame(
                shop_excluded_race=self._shop_excluded_race,
                shop_excluded_count=self._shop_excluded_count,
                shop_full_tribes=self._shop_full_tribes,
                patch_dir=self._patch_dir,
            )
        self._state = self._game.initial_state()
        self._rl_pending = None
        self._rl_place_budget_pending = False
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
        from src.envs.minibg.invariants import assert_action_in_legal_mask

        assert_action_in_legal_mask(
            self._state,
            action_int,
            legal_mask,
            where="MiniBGEnv.step",
            rl_pending=self._rl_pending is not None,
        )

        acting_idx = self._state.current_player_index
        acting_token = PLAYER_TOKENS[acting_idx]
        prev_round = self._state.round_number
        prev_done = self._state.done
        prev_hp = (
            self._state.players[0].health,
            self._state.players[1].health,
        )

        from src.envs.minibg.invariants import raise_illegal_env_action

        try:
            if is_swap_board(action_int):
                i = swap_adj_index_from_env_action(action_int)
                self._state = self._game.swap_board_adjacent(self._state, acting_idx, i)
            elif self._rl_pending is not None:
                if is_apply_effect_skip(action_int):
                    if not self._rl_pending.can_skip_second_adjacent():
                        raise_illegal_env_action(
                            self._state,
                            where="MiniBGEnv.step.rl_pending",
                            action=action_int,
                            mask=legal_mask,
                            detail="APPLY_EFFECT_SKIP not allowed",
                            rl_pending=True,
                        )
                    self._apply_rl_effect_skip()
                elif is_target_board(action_int):
                    self._apply_rl_effect_pick(target_board_slot(action_int))
                else:
                    raise_illegal_env_action(
                        self._state,
                        where="MiniBGEnv.step.rl_pending",
                        action=action_int,
                        mask=legal_mask,
                        detail="expected TARGET_BOARD or SKIP during rl_pending",
                        rl_pending=True,
                    )
            elif is_place(action_int):
                if not self._try_place_hand_rl(place_slot(action_int)):
                    raise_illegal_env_action(
                        self._state,
                        where="MiniBGEnv.step.place",
                        action=action_int,
                        mask=legal_mask,
                        detail="PLACE failed",
                        rl_pending=False,
                    )
            else:
                game_action = env_action_to_game_action(action_int)
                self._state = self._game.apply_action(self._state, game_action)
        except ValueError as exc:
            raise_illegal_env_action(
                self._state,
                where="MiniBGEnv.step.apply",
                action=action_int,
                mask=legal_mask,
                detail=str(exc),
                rl_pending=self._rl_pending is not None,
            )

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

        if self._rl_pending is not None:
            eligible = self._rl_pending.eligible_on_board_live(player.board)
            for tgt in eligible:
                out.append(StructAction(StructActionType.APPLY_EFFECT, (tgt,)))
            if self._rl_pending.can_skip_second_adjacent():
                out.append(StructAction(StructActionType.APPLY_EFFECT_SKIP))
            return out

        if player.pending_choice is not None:
            pc = player.pending_choice
            if pc.kind == PendingChoiceKind.TRANSFORM_SHOP_MINION:
                for slot in range(MAX_SHOP_SLOTS):
                    if (int(GameAction.BUY_SLOT_0) + slot) in legal_game:
                        out.append(StructAction(StructActionType.BUY, (slot,)))
                return out
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

        if (
            self._rl_pending is None
            and player.placed_minion_pending_after is None
        ):
            if int(GameAction.FINISH) in legal_game:
                out.append(StructAction(StructActionType.COMPLETE_TURN))
            if int(GameAction.FINISH_FREEZE_SHOP) in legal_game:
                out.append(StructAction(StructActionType.COMPLETE_TURN_FREEZE_SHOP))

        if not out and not legal_game:
            from src.envs.minibg.invariants import assert_shop_has_legal_actions

            assert_shop_has_legal_actions(
                self._state,
                [],
                where="MiniBGEnv.legal_structured_actions",
            )
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

        from src.envs.minibg.invariants import raise_illegal_env_action

        if action not in legal:
            raise_illegal_env_action(
                self._state,
                where="MiniBGEnv.step_structured",
                structured_action=repr(action),
                detail="not in legal_structured_actions",
                rl_pending=self._rl_pending is not None,
            )

        if action.type in (
            StructActionType.COMPLETE_TURN,
            StructActionType.COMPLETE_TURN_FREEZE_SHOP,
        ):
            if board_perm is None:
                raise_illegal_env_action(
                    self._state,
                    where="MiniBGEnv.step_structured",
                    structured_action=repr(action),
                    detail="COMPLETE_TURN requires board_perm",
                    rl_pending=self._rl_pending is not None,
                )
            try:
                validate_board_perm(tuple(board_perm))
            except ValueError as exc:
                raise_illegal_env_action(
                    self._state,
                    where="MiniBGEnv.step_structured",
                    structured_action=repr(action),
                    detail=f"invalid board_perm: {exc}",
                    rl_pending=self._rl_pending is not None,
                )
        elif board_perm is not None:
            raise_illegal_env_action(
                self._state,
                where="MiniBGEnv.step_structured",
                structured_action=repr(action),
                detail="board_perm only allowed for COMPLETE_TURN",
                rl_pending=self._rl_pending is not None,
            )

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
            elif action.type == StructActionType.APPLY_EFFECT:
                if self._rl_pending is None:
                    raise_illegal_env_action(
                        self._state,
                        where="MiniBGEnv.step_structured.apply",
                        structured_action=repr(action),
                        detail="APPLY_EFFECT without rl_pending",
                    )
                self._apply_rl_effect_pick(action.args[0])
            elif action.type == StructActionType.APPLY_EFFECT_SKIP:
                if self._rl_pending is None:
                    raise_illegal_env_action(
                        self._state,
                        where="MiniBGEnv.step_structured.apply",
                        structured_action=repr(action),
                        detail="APPLY_EFFECT_SKIP without rl_pending",
                    )
                self._apply_rl_effect_skip()
            elif action.type == StructActionType.PLACE:
                if not self._try_place_hand_rl(action.args[0]):
                    raise_illegal_env_action(
                        self._state,
                        where="MiniBGEnv.step_structured.place",
                        structured_action=repr(action),
                        detail="PLACE failed",
                    )
            else:
                ga = self._struct_action_to_game_action(action)
                self._state = self._game.apply_action(self._state, ga)
        except ValueError as exc:
            raise_illegal_env_action(
                self._state,
                where="MiniBGEnv.step_structured.apply",
                structured_action=repr(action),
                detail=str(exc),
                rl_pending=self._rl_pending is not None,
            )

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
        if action.type == StructActionType.APPLY_EFFECT:
            return int(GameAction.TARGET_BOARD_0) + action.args[0]
        raise ValueError(f"not a shop-phase structured action: {action}")

    @property
    def legal_actions_mask(self) -> np.ndarray:
        if self._state is None:
            return np.zeros(NUM_ENV_ACTIONS, dtype=bool)
        mask = np.zeros(NUM_ENV_ACTIONS, dtype=bool)
        if self._state.done:
            return mask

        player = self._state.players[self._state.current_player_index]

        from .action_map import A_APPLY_EFFECT_SKIP, A_TARGET_BOARD_BASE

        if self._rl_pending is not None:
            for i in self._rl_pending.eligible_on_board_live(player.board):
                mask[A_TARGET_BOARD_BASE + i] = True
            if self._rl_pending.can_skip_second_adjacent():
                mask[A_APPLY_EFFECT_SKIP] = True
            return mask

        if (
            player.phase == PlayerPhase.SHOP
            and player.pending_choice is None
        ):
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

        if self._rl_pending is None:
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

        for i in range(BOARD_SIZE):
            if int(GameAction.TARGET_BOARD_0) + i in legal_game:
                mask[A_TARGET_BOARD_BASE + i] = True

        if player.placed_minion_pending_after is None:
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
            rl_pending=self._rl_pending,
            patch=self._game._patch,
        )

    def _try_place_hand_rl(self, hand_slot: int) -> bool:
        assert self._state is not None
        if self._rl_pending is not None:
            return False
        player = self._state.players[self._state.current_player_index]
        legal = set(int(a) for a in self._game.legal_actions(self._state))
        if int(GameAction.PLACE_HAND_0) + hand_slot not in legal:
            return False
        if player.hand[hand_slot] is None:
            return False
        self._rl_place_budget_pending = True
        plan = open_rl_place_plan(player, hand_slot)
        if plan is None:
            commit_simple_place_from_hand(
                player,
                hand_slot,
                self._state.shop_excluded_race,
                board_size=BOARD_SIZE,
                triggers=self._game._shop_triggers,
                rng=self._game._rng,
            )
            self._finish_rl_place_after_effects()
            return True
        self._rl_pending = plan
        return True

    def _apply_rl_effect_pick(self, board_idx: int) -> None:
        assert self._state is not None and self._rl_pending is not None
        player = self._state.players[self._state.current_player_index]
        self._rl_pending.record_pick_live(player.board, board_idx)
        if self._rl_pending.is_complete():
            self._commit_rl_place_plan()

    def _apply_rl_effect_skip(self) -> None:
        assert self._state is not None and self._rl_pending is not None
        self._rl_pending.record_skip_second()
        if self._rl_pending.is_complete():
            self._commit_rl_place_plan()

    def _commit_rl_place_plan(self) -> None:
        assert self._state is not None and self._rl_pending is not None
        idx = self._state.current_player_index
        plan = self._rl_pending
        self._rl_pending = None
        self._state = commit_rl_place_plan(
            self._state,
            idx,
            plan,
            board_size=BOARD_SIZE,
            shop_excluded_race=self._state.shop_excluded_race,
            triggers=self._game._shop_triggers,
            rng=self._game._rng,
            reorder_board=self._game.reorder_board,
        )
        self._finish_rl_place_after_effects()

    def _finish_rl_place_after_effects(self) -> None:
        assert self._state is not None
        player = self._state.players[self._state.current_player_index]
        ref = player.placed_minion_pending_after
        if (
            ref is not None
            and ref in player.board
            and player.pending_choice is None
        ):
            self._game._shop_triggers.fire_after_friendly_minion_placed(player, ref)
        player.placed_minion_board_index = None
        player.placed_minion_pending_after = None
        flush_triple_reward_queue_if_idle(
            player,
            self._state.shop_excluded_race,
            rng=self._game._rng,
            patch=self._game._patch,
        )
        if self._rl_place_budget_pending:
            player.shop_actions_used += 1
            self._rl_place_budget_pending = False

    @staticmethod
    def _minion_repr(m: Minion) -> str:
        s = "S" if m.has_shield else ""
        return f"{m.card_id}:{m.bonus_attack}+{m.bonus_health}{s}"


__all__ = ["MiniBGEnv", "INVALID_ACTION_REWARD"]
