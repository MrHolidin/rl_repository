from __future__ import annotations

from copy import copy
from typing import List, Optional, Sequence

import numpy as np

from src.bg_lobby import round as bg_lobby_round
from src.bg_lobby.shared_pool import SharedCardPool, build_initial_shared_pool
from src.bg_lobby.shop_order import sample_shop_turn_order
from src.bg_player_turn import PlayerTurnContext, PlayerTurnEngine
from src.games.turn_based_game import Action as ActionType
from src.games.turn_based_game import TurnBasedGame

from .actions import (
    BOARD_SIZE,
    COMBAT_BOARD_MAX,
    DAMAGE_CAP,
    HAND_SIZE,
    LEVEL_UP_COSTS,
    LEVEL_UP_DISCOUNT_PER_ROUND,
    MAX_ROUNDS,
    MAX_SHOP_SLOTS,
    MAX_TIER,
    Action,
    gold_for_round,
    STARTING_HEALTH,
    STARTING_TIER,
)
from src.bg_recruitment import place as recruitment_place
from src.bg_recruitment import shop as recruitment_shop
from src.bg_recruitment import triples as recruitment_triples
from src.bg_recruitment import discover as recruitment_discover
from src.bg_recruitment.shop_triggers import ShopTriggers

from . import board_order as minibg_board_order
from .state import (
    MiniBGState,
    Minion,
    PendingChoice,
    PlayerPhase,
    PlayerState,
    Race,
    ROTATION_SHOP_TRIBES,
)


PLAYER_TOKENS = (1, -1)


class MiniBGGame(TurnBasedGame[MiniBGState]):
    def __init__(
        self,
        seed: Optional[int] = None,
        *,
        shop_excluded_race: Optional[Race] = None,
        shop_full_tribes: bool = False,
        use_shared_pool: bool = False,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._shop_full_tribes = shop_full_tribes
        self._use_shared_pool = use_shared_pool
        self._shop_excluded_race_fixed = shop_excluded_race
        self._shop_triggers = ShopTriggers(
            self._rng, on_triples=recruitment_triples.resolve_triples_loop
        )
        self._player_turn = PlayerTurnEngine()

    def _turn_ctx(self, state: MiniBGState) -> PlayerTurnContext:
        return PlayerTurnContext(
            rng=self._rng,
            triggers=self._shop_triggers,
            shop_excluded_race=state.shop_excluded_race,
            shared_pool=state.shared_pool,
        )

    def _pick_shop_excluded_race(self) -> Optional[Race]:
        if self._shop_excluded_race_fixed is not None:
            return self._shop_excluded_race_fixed
        if self._shop_full_tribes:
            return None
        i = int(self._rng.integers(0, len(ROTATION_SHOP_TRIBES)))
        return ROTATION_SHOP_TRIBES[i]

    def initial_state(self) -> MiniBGState:
        initiative_player = int(self._rng.integers(0, 2))
        shop_excluded = self._pick_shop_excluded_race()
        shared_pool = (
            build_initial_shared_pool(shop_excluded)
            if self._use_shared_pool
            else None
        )
        players = (
            self._fresh_player(
                round_number=1,
                shop_excluded_race=shop_excluded,
                shared_pool=shared_pool,
            ),
            self._fresh_player(
                round_number=1,
                shop_excluded_race=shop_excluded,
                shared_pool=shared_pool,
            ),
        )
        order = sample_shop_turn_order(self._rng, 2)
        return MiniBGState(
            players=players,
            round_number=1,
            current_player_index=order[0],
            initiative_player=initiative_player,
            winner=None,
            done=False,
            shop_excluded_race=shop_excluded,
            shop_turn_order=(order[0], order[1]),
            shared_pool=shared_pool,
        )

    def current_player(self, state: MiniBGState) -> int:
        return PLAYER_TOKENS[state.current_player_index]

    def is_terminal(self, state: MiniBGState) -> bool:
        return state.done

    def winner(self, state: MiniBGState) -> Optional[int]:
        return state.winner

    def legal_actions(self, state: MiniBGState) -> Sequence[ActionType]:
        if state.done:
            return []
        player = state.players[state.current_player_index]
        if player.pending_choice is not None:
            pc = player.pending_choice
            if recruitment_discover.is_hand_discover_kind(pc.kind):
                if not recruitment_triples.hand_has_free_slot(player):
                    from src.envs.minibg.invariants import assert_shop_has_legal_actions

                    assert_shop_has_legal_actions(
                        state,
                        [],
                        where="game.legal_actions.hand_discover_full_hand",
                    )
        return self._player_turn.legal_actions(player)

    def apply_action(self, state: MiniBGState, action: ActionType) -> MiniBGState:
        if state.done:
            raise ValueError("Cannot apply action in terminal state")

        action_int = int(action)
        legal = self.legal_actions(state)
        if action_int not in legal:
            raise ValueError(
                f"Illegal action {action_int} in current state "
                f"(player={state.current_player_index}, "
                f"phase={state.players[state.current_player_index].phase.name}, "
                f"gold={state.players[state.current_player_index].gold})"
            )

        new_state = self._copy_state(state)
        idx = new_state.current_player_index
        player = new_state.players[idx]
        ctx = self._turn_ctx(new_state)

        if action_int == int(Action.FINISH):
            self._player_turn.end_turn(player, freeze_shop=False)
            self._after_player_finished(new_state, idx)
            return new_state

        if action_int == int(Action.FINISH_FREEZE_SHOP):
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
        state: MiniBGState,
        player_idx: int,
        perm: Sequence[int],
    ) -> MiniBGState:
        return minibg_board_order.reorder_board(
            state,
            player_idx,
            perm,
            board_size=BOARD_SIZE,
            copy_state=self._copy_state,
        )

    def swap_board_adjacent(
        self,
        state: MiniBGState,
        player_idx: int,
        i: int,
    ) -> MiniBGState:
        return minibg_board_order.swap_board_adjacent(
            state, player_idx, i, copy_state=self._copy_state
        )

    def _after_player_finished(self, state: MiniBGState, idx: int) -> None:
        bg_lobby_round.after_player_finished(
            state,
            idx,
            fire_on_turn_end=self._shop_triggers.fire_on_turn_end,
            resolve_battle_and_advance=self._resolve_battle_and_advance,
        )

    # Backward-compatible hooks for tests / bots (implementation in bg_recruitment).
    _hand_has_free_slot = staticmethod(recruitment_triples.hand_has_free_slot)
    _merge_three_non_golden_into_golden = staticmethod(
        recruitment_triples.merge_three_non_golden_into_golden
    )
    merge_magnetic_inplace = staticmethod(recruitment_place.merge_magnetic_inplace)

    def _fire_on_place(
        self,
        placed: Minion,
        player: PlayerState,
        shop_excluded_race: Optional[Race],
    ) -> None:
        self._shop_triggers.fire_on_place(placed, player, shop_excluded_race)

    def _fire_on_turn_end(self, player: PlayerState) -> None:
        self._shop_triggers.fire_on_turn_end(player)

    def _refresh_shop(
        self,
        player: PlayerState,
        shop_excluded_race: Optional[Race],
        *,
        shared_pool: Optional[SharedCardPool] = None,
    ) -> None:
        recruitment_shop.refresh_shop(
            player,
            shop_excluded_race,
            rng=self._rng,
            shared_pool=shared_pool,
        )

    def _refresh_shop_fill_empty_slots(
        self,
        player: PlayerState,
        shop_excluded_race: Optional[Race],
        *,
        shared_pool: Optional[SharedCardPool] = None,
    ) -> None:
        recruitment_shop.refresh_shop_fill_empty_slots(
            player,
            shop_excluded_race,
            rng=self._rng,
            shared_pool=shared_pool,
        )

    def _resolve_battle_and_advance(self, state: MiniBGState) -> None:
        pool = state.shared_pool

        def refresh_shop(p: PlayerState, exc: Optional[Race]) -> None:
            self._refresh_shop(p, exc, shared_pool=pool)

        def refresh_fill(p: PlayerState, exc: Optional[Race]) -> None:
            self._refresh_shop_fill_empty_slots(p, exc, shared_pool=pool)

        bg_lobby_round.resolve_battle_and_advance(
            state,
            rng=self._rng,
            combat_board_max=COMBAT_BOARD_MAX,
            damage_cap=DAMAGE_CAP,
            board_size=BOARD_SIZE,
            max_rounds=MAX_ROUNDS,
            max_tier=MAX_TIER,
            level_up_discount_per_round=LEVEL_UP_DISCOUNT_PER_ROUND,
            fire_on_turn_start=self._shop_triggers.fire_on_turn_start,
            refresh_shop=refresh_shop,
            refresh_shop_fill_empty_slots=refresh_fill,
        )

    def _fresh_player(
        self,
        round_number: int,
        shop_excluded_race: Optional[Race],
        *,
        shared_pool: Optional[SharedCardPool] = None,
    ) -> PlayerState:
        player = PlayerState(
            health=STARTING_HEALTH,
            hero_damage_taken_total=0,
            gold=gold_for_round(round_number),
            tavern_tier=STARTING_TIER,
            next_tier_up_cost=LEVEL_UP_COSTS[STARTING_TIER],
            board=[],
            shop=[None for _ in range(MAX_SHOP_SLOTS)],
            hand=[None for _ in range(HAND_SIZE)],
            phase=PlayerPhase.SHOP,
            shop_actions_used=0,
            pending_choice=None,
            placed_minion_board_index=None,
            placed_minion_pending_after=None,
        )
        self._refresh_shop(player, shop_excluded_race, shared_pool=shared_pool)
        return player

    @staticmethod
    def _copy_state(state: MiniBGState) -> MiniBGState:
        new_players = (
            MiniBGGame._copy_player(state.players[0]),
            MiniBGGame._copy_player(state.players[1]),
        )
        return MiniBGState(
            players=new_players,
            round_number=state.round_number,
            current_player_index=state.current_player_index,
            initiative_player=state.initiative_player,
            winner=state.winner,
            done=state.done,
            shop_excluded_race=state.shop_excluded_race,
            shop_turn_order=state.shop_turn_order,
            shared_pool=(
                state.shared_pool.copy()
                if state.shared_pool is not None
                else None
            ),
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


__all__ = ["MiniBGGame", "PLAYER_TOKENS"]
