from __future__ import annotations

from copy import copy
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.bg_lobby import eight_player as bg_lobby_eight
from src.bg_lobby.shared_pool import SharedCardPool, build_initial_shared_pool
from src.bg_lobby.shop_order import sample_shop_turn_order
from src.bg_player_turn import PlayerTurnContext, PlayerTurnEngine
from src.games.turn_based_game import Action as ActionType
from src.games.turn_based_game import TurnBasedGame

from src.bg_recruitment import place as recruitment_place
from src.bg_recruitment import shop as recruitment_shop
from src.bg_recruitment import triples as recruitment_triples
from src.bg_recruitment import discover as recruitment_discover
from src.bg_recruitment.shop_triggers import ShopTriggers

from src.bg_catalog.cards import normalize_shop_excluded_races
from src.bg_catalog.patch_context import PatchContext, load_patch_context

from . import actions as bglike_actions
from .state import (
    BGLikeState,
    Minion,
    PendingChoice,
    PlayerPhase,
    PlayerState,
    Race,
)


def _resolve_patch(
    patch: Optional[PatchContext],
    patch_dir: Optional[str],
) -> PatchContext:
    if patch is not None and patch_dir is not None:
        raise ValueError("pass patch or patch_dir, not both")
    if patch_dir is not None:
        return load_patch_context(patch_dir)
    if patch is not None:
        return patch
    raise ValueError("BGLikeGame requires patch or patch_dir")


class BGLikeGame(TurnBasedGame[BGLikeState]):
    def __init__(
        self,
        seed: Optional[int] = None,
        *,
        shop_excluded_race: Optional[Race | Tuple[Race, ...]] = None,
        shop_excluded_count: Optional[int] = None,
        shop_full_tribes: bool = False,
        patch: Optional[PatchContext] = None,
        patch_dir: Optional[str] = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._shop_full_tribes = shop_full_tribes
        self._patch = _resolve_patch(patch, patch_dir)
        self._shop_excluded_race_fixed = (
            tuple(normalize_shop_excluded_races(shop_excluded_race))
            if shop_excluded_race is not None
            else None
        )
        self._shop_excluded_count = (
            self._patch.meta.rotation_excluded_count
            if shop_excluded_count is None
            else int(shop_excluded_count)
        )
        self._shop_triggers = ShopTriggers(self._rng, patch=self._patch)
        self._player_turn = PlayerTurnEngine(bglike_actions)

    def _turn_ctx(self, state: BGLikeState) -> PlayerTurnContext:
        return PlayerTurnContext(
            rng=self._rng,
            triggers=self._shop_triggers,
            shop_excluded_race=state.shop_excluded_race,
            shared_pool=state.shared_pool,
            patch=self._patch,
        )

    def _pick_shop_excluded_race(self) -> Optional[Tuple[Race, ...]]:
        if self._shop_excluded_race_fixed is not None:
            return self._shop_excluded_race_fixed
        if self._shop_full_tribes:
            return None
        tribes = self._patch.meta.rotation_tribes
        max_excluded = max(0, len(tribes) - 1)
        excluded_count = max(0, min(int(self._shop_excluded_count), max_excluded))
        if excluded_count <= 0:
            return None
        picks = self._rng.choice(len(tribes), size=excluded_count, replace=False)
        return tuple(tribes[int(i)] for i in picks)

    def initial_state(self) -> BGLikeState:
        n = bglike_actions.NUM_PLAYERS
        shop_excluded = self._pick_shop_excluded_race()
        shared_pool = build_initial_shared_pool(
            shop_excluded,
            patch=self._patch,
        )
        players = tuple(
            self._fresh_player(round_number=1, shop_excluded_race=shop_excluded, shared_pool=shared_pool)
            for _ in range(n)
        )
        alive = tuple(range(n))
        order = sample_shop_turn_order(self._rng, len(alive))
        return BGLikeState(
            players=players,
            alive=alive,
            round_number=1,
            combat_round=0,
            full_lobby_cycle_round=0,
            current_player_index=order[0],
            shop_turn_order=order,
            recent_opponents=tuple(() for _ in range(n)),
            eliminated=(),
            pairings=(),
            initiative_player=int(self._rng.integers(0, n)),
            winner=None,
            done=False,
            shop_excluded_race=shop_excluded,
            shared_pool=shared_pool,
            patch_build=self._patch.build,
        )

    def current_player(self, state: BGLikeState) -> int:
        return state.current_player_index

    def is_terminal(self, state: BGLikeState) -> bool:
        return state.done

    def winner(self, state: BGLikeState) -> Optional[int]:
        return state.winner

    def legal_actions(self, state: BGLikeState) -> Sequence[ActionType]:
        if state.done:
            return []
        player = state.players[state.current_player_index]
        if player.pending_choice is not None:
            pc = player.pending_choice
            if recruitment_discover.is_hand_discover_kind(pc.kind):
                if not recruitment_triples.hand_has_free_slot(player):
                    raise RuntimeError(
                        "hand discover with full hand (legal mask bug)"
                    )
        return self._player_turn.legal_actions(player)

    def apply_action(self, state: BGLikeState, action: ActionType) -> BGLikeState:
        if state.done:
            raise ValueError("Cannot apply action in terminal state")

        action_int = int(action)
        legal = self.legal_actions(state)
        if action_int not in legal:
            raise ValueError(
                f"Illegal action {action_int} "
                f"(player={state.current_player_index}, "
                f"phase={state.players[state.current_player_index].phase.name})"
            )

        new_state = self._copy_state(state)
        idx = new_state.current_player_index
        player = new_state.players[idx]
        ctx = self._turn_ctx(new_state)

        if action_int == int(bglike_actions.Action.FINISH):
            self._player_turn.end_turn(player, freeze_shop=False)
            self._after_player_finished(new_state, idx)
            return new_state

        if action_int == int(bglike_actions.Action.FINISH_FREEZE_SHOP):
            self._player_turn.end_turn(player, freeze_shop=True)
            self._after_player_finished(new_state, idx)
            return new_state

        consumes_budget = self._player_turn.apply(
            player,
            action_int,
            ctx,
            shop_excluded_race=new_state.shop_excluded_race,
        )
        if consumes_budget:
            player.shop_actions_used += 1

        return new_state

    def reorder_board(
        self,
        state: BGLikeState,
        player_idx: int,
        perm: Sequence[int],
    ) -> BGLikeState:
        from src.envs.minibg.board_order import reorder_board

        return reorder_board(
            state,
            player_idx,
            perm,
            board_size=bglike_actions.BOARD_SIZE,
            copy_state=self._copy_state,
        )

    def swap_board_adjacent(
        self,
        state: BGLikeState,
        player_idx: int,
        i: int,
    ) -> BGLikeState:
        from src.envs.minibg.board_order import swap_board_adjacent

        return swap_board_adjacent(
            state,
            player_idx,
            i,
            copy_state=self._copy_state,
        )

    def _after_player_finished(self, state: BGLikeState, idx: int) -> None:
        bg_lobby_eight.after_player_finished(
            state,
            idx,
            fire_on_turn_end=self._shop_triggers.fire_on_turn_end,
            resolve_combat_round=self._resolve_combat_round,
        )

    def _refresh_shop(
        self,
        player: PlayerState,
        shop_excluded_race: Optional[Race],
        *,
        shared_pool: Optional[SharedCardPool],
    ) -> None:
        recruitment_shop.refresh_shop(
            player,
            shop_excluded_race,
            rng=self._rng,
            shared_pool=shared_pool,
            frozen_slots=player.shop_frozen,
            patch=self._patch,
        )

    def _refresh_shop_fill_empty_slots(
        self,
        player: PlayerState,
        shop_excluded_race: Optional[Race],
        *,
        shared_pool: Optional[SharedCardPool],
    ) -> None:
        recruitment_shop.refresh_shop_fill_empty_slots(
            player,
            shop_excluded_race,
            rng=self._rng,
            shared_pool=shared_pool,
            frozen_slots=player.shop_frozen,
            patch=self._patch,
        )

    def _resolve_combat_round(self, state: BGLikeState) -> None:
        pool = state.shared_pool

        def refresh_shop(p: PlayerState, exc: Optional[Race]) -> None:
            self._refresh_shop(p, exc, shared_pool=pool)

        def refresh_fill(p: PlayerState, exc: Optional[Race]) -> None:
            self._refresh_shop_fill_empty_slots(p, exc, shared_pool=pool)

        bg_lobby_eight.resolve_combat_round(
            state,
            rng=self._rng,
            combat_board_max=bglike_actions.COMBAT_BOARD_MAX,
            damage_cap=bglike_actions.DAMAGE_CAP,
            board_size=bglike_actions.BOARD_SIZE,
            max_tier=bglike_actions.MAX_TIER,
            level_up_discount_per_round=bglike_actions.LEVEL_UP_DISCOUNT_PER_ROUND,
            max_rounds=bglike_actions.MAX_ROUNDS,
            fire_on_turn_start=self._shop_triggers.fire_on_turn_start,
            refresh_shop=refresh_shop,
            refresh_shop_fill_empty_slots=refresh_fill,
        )

    def _fresh_player(
        self,
        round_number: int,
        shop_excluded_race: Optional[Race],
        *,
        shared_pool: SharedCardPool,
    ) -> PlayerState:
        player = PlayerState(
            health=bglike_actions.STARTING_HEALTH,
            hero_damage_taken_total=0,
            gold=bglike_actions.gold_for_round(round_number),
            tavern_tier=bglike_actions.STARTING_TIER,
            next_tier_up_cost=bglike_actions.LEVEL_UP_COSTS[bglike_actions.STARTING_TIER],
            board=[],
            shop=[None for _ in range(bglike_actions.MAX_SHOP_SLOTS)],
            hand=[None for _ in range(bglike_actions.HAND_SIZE)],
            phase=PlayerPhase.SHOP,
            shop_actions_used=0,
            pending_choice=None,
            placed_minion_board_index=None,
            placed_minion_pending_after=None,
        )
        self._refresh_shop(player, shop_excluded_race, shared_pool=shared_pool)
        return player

    def _copy_state(self, state: BGLikeState) -> BGLikeState:
        new_players = tuple(
            BGLikeGame._copy_player(p) for p in state.players
        )
        return BGLikeState(
            players=new_players,
            alive=state.alive,
            round_number=state.round_number,
            combat_round=state.combat_round,
            full_lobby_cycle_round=state.full_lobby_cycle_round,
            current_player_index=state.current_player_index,
            shop_turn_order=state.shop_turn_order,
            recent_opponents=state.recent_opponents,
            eliminated=state.eliminated,
            pairings=state.pairings,
            initiative_player=state.initiative_player,
            winner=state.winner,
            done=state.done,
            shop_excluded_race=state.shop_excluded_race,
            shared_pool=state.shared_pool.copy() if state.shared_pool else None,
            patch_build=state.patch_build,
        )

    @staticmethod
    def _copy_player(p: PlayerState) -> PlayerState:
        new_board = [copy(m) for m in p.board]
        remapped_pending: Optional[Minion] = None
        pend = p.placed_minion_pending_after
        if pend is not None:
            try:
                i = p.board.index(pend)
            except ValueError:
                pass
            else:
                if 0 <= i < len(new_board):
                    remapped_pending = new_board[i]
        placed_idx: Optional[int] = None
        if remapped_pending is not None:
            try:
                placed_idx = new_board.index(remapped_pending)
            except ValueError:
                placed_idx = None
        return PlayerState(
            health=p.health,
            hero_damage_taken_total=p.hero_damage_taken_total,
            gold=p.gold,
            tavern_tier=p.tavern_tier,
            next_tier_up_cost=p.next_tier_up_cost,
            board=new_board,
            shop=[copy(m) if m is not None else None for m in p.shop],
            hand=[copy(m) if m is not None else None for m in p.hand],
            phase=p.phase,
            shop_actions_used=p.shop_actions_used,
            shop_freeze_next_round=p.shop_freeze_next_round,
            pending_choice=(
                PendingChoice(
                    p.pending_choice.kind,
                    p.pending_choice.options,
                    p.pending_choice.extra_modals_after,
                    p.pending_choice.options_pool_reserved,
                    p.pending_choice.transform_board_idx,
                )
                if p.pending_choice is not None
                else None
            ),
            placed_minion_board_index=placed_idx,
            placed_minion_pending_after=remapped_pending,
            triple_reward_discover_pending=p.triple_reward_discover_pending,
            triple_reward_spell_tier=p.triple_reward_spell_tier,
            pogo_hoppers_played=p.pogo_hoppers_played,
        )


__all__ = ["BGLikeGame"]
