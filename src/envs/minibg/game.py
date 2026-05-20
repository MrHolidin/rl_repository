from __future__ import annotations

from copy import copy
from typing import List, Optional, Sequence

import numpy as np

from src.games.turn_based_game import Action as ActionType
from src.games.turn_based_game import TurnBasedGame

from .actions import (
    BOARD_SIZE,
    COMBAT_BOARD_MAX,
    BUY_COST,
    DAMAGE_CAP,
    HAND_SIZE,
    LEVEL_UP_COSTS,
    LEVEL_UP_DISCOUNT_PER_ROUND,
    MAX_ROUNDS,
    MAX_SHOP_ACTIONS,
    MAX_SHOP_SLOTS,
    MAX_TIER,
    ROLL_COST,
    STARTING_HEALTH,
    STARTING_TIER,
    Action,
    discover_pick_index,
    gold_for_round,
    is_discover_pick_game_action,
    is_magnet_game_action,
    magnet_hand_board_from_game_action,
    shop_offers_count,
)
from src.bg_recruitment import discover as recruitment_discover
from src.bg_recruitment import economy as recruitment_economy
from src.bg_recruitment import place as recruitment_place
from src.bg_recruitment import shop as recruitment_shop
from src.bg_recruitment import triples as recruitment_triples
from src.bg_recruitment.shop_triggers import ShopTriggers

from . import board_order as minibg_board_order
from . import round as minibg_round
from .state import (
    MiniBGState,
    Minion,
    PendingChoice,
    PendingChoiceKind,
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
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._shop_full_tribes = shop_full_tribes
        self._shop_excluded_race_fixed = shop_excluded_race
        self._shop_triggers = ShopTriggers(
            self._rng, on_triples=recruitment_triples.resolve_triples_loop
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
        players = (
            self._fresh_player(round_number=1, shop_excluded_race=shop_excluded),
            self._fresh_player(round_number=1, shop_excluded_race=shop_excluded),
        )
        return MiniBGState(
            players=players,
            round_number=1,
            current_player_index=0,
            initiative_player=initiative_player,
            winner=None,
            done=False,
            shop_excluded_race=shop_excluded,
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
        if player.phase == PlayerPhase.DONE:
            return []

        # Shop phase — pending Discover / Adapt blocks other actions.
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
            return [
                int(Action.DISCOVER_PICK_0),
                int(Action.DISCOVER_PICK_1),
                int(Action.DISCOVER_PICK_2),
            ]

        actions: List[int] = []

        # Shop phase.
        can_act = player.shop_actions_used < MAX_SHOP_ACTIONS
        hand_full = sum(1 for m in player.hand if m is not None) >= HAND_SIZE
        board_full = len(player.board) >= BOARD_SIZE

        if can_act:
            if not hand_full:
                n_offers = shop_offers_count(player.tavern_tier)
                for slot in range(min(n_offers, len(player.shop))):
                    if (
                        player.shop[slot] is not None
                        and player.gold >= BUY_COST
                    ):
                        actions.append(int(Action.BUY_SLOT_0) + slot)

            for pos in range(BOARD_SIZE):
                if pos < len(player.board):
                    actions.append(int(Action.SELL_BOARD_0) + pos)

            for h in range(HAND_SIZE):
                hm = player.hand[h]
                if hm is None:
                    continue
                if recruitment_triples.is_triple_reward_discover_spell(hm):
                    actions.append(int(Action.PLACE_HAND_0) + h)
                    continue
                if board_full:
                    continue
                bc_mult = ShopTriggers.battlecry_multiplier(player.board)
                needs = recruitment_discover.discover_cards_to_receive(hm, bc_mult)
                free = sum(1 for s in player.hand if s is None) + 1
                if needs > free:
                    continue
                actions.append(int(Action.PLACE_HAND_0) + h)

            for h in range(HAND_SIZE):
                hm = player.hand[h]
                if hm is None or not recruitment_place.hand_minion_can_magnetize(hm):
                    continue
                for b in range(len(player.board)):
                    if recruitment_place.is_mech(player.board[b]):
                        actions.append(
                            int(Action.MAGNET_HAND_0_BOARD_0) + h * BOARD_SIZE + b
                        )

            if player.gold >= ROLL_COST:
                actions.append(int(Action.ROLL))

            if (
                player.tavern_tier < MAX_TIER
                and player.gold >= player.next_tier_up_cost
            ):
                actions.append(int(Action.LEVEL_UP))

        actions.append(int(Action.FINISH))
        actions.append(int(Action.FINISH_FREEZE_SHOP))
        return actions

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

        if action_int == int(Action.FINISH):
            player.shop_freeze_next_round = False
            self._submit_order(player)
            self._after_player_finished(new_state, idx)
            return new_state

        if action_int == int(Action.FINISH_FREEZE_SHOP):
            player.shop_freeze_next_round = True
            self._submit_order(player)
            self._after_player_finished(new_state, idx)
            return new_state

        consumes_budget = False

        if player.pending_choice is not None:
            if not is_discover_pick_game_action(action_int):
                raise ValueError(
                    f"Expected DISCOVER_PICK_* while pending_choice, got {action_int}"
                )
            recruitment_discover.resolve_discover_pick(
                player,
                discover_pick_index(action_int),
                new_state.shop_excluded_race,
                rng=self._rng,
                on_after_placed=self._shop_triggers.fire_after_friendly_minion_placed,
            )
            consumes_budget = (
                player.pending_choice is None
                and player.placed_minion_pending_after is None
            )
        elif int(Action.BUY_SLOT_0) <= action_int < int(Action.BUY_SLOT_0) + MAX_SHOP_SLOTS:
            self._do_buy(player, action_int - int(Action.BUY_SLOT_0))
            consumes_budget = True
        elif int(Action.SELL_BOARD_0) <= action_int < int(Action.SELL_BOARD_0) + BOARD_SIZE:
            self._do_sell(player, action_int - int(Action.SELL_BOARD_0))
            consumes_budget = True
        elif action_int == int(Action.ROLL):
            self._do_roll(player, new_state.shop_excluded_race)
            consumes_budget = True
        elif action_int == int(Action.LEVEL_UP):
            self._do_level_up(player, new_state.shop_excluded_race)
            consumes_budget = True
        elif int(Action.PLACE_HAND_0) <= action_int < int(Action.PLACE_HAND_0) + HAND_SIZE:
            recruitment_place.place_from_hand(
                player,
                action_int - int(Action.PLACE_HAND_0),
                new_state.shop_excluded_race,
                board_size=BOARD_SIZE,
                triggers=self._shop_triggers,
                rng=self._rng,
            )
            consumes_budget = (
                player.pending_choice is None
                and player.placed_minion_pending_after is None
            )
        elif is_magnet_game_action(action_int):
            h, b = magnet_hand_board_from_game_action(action_int)
            recruitment_place.magnet_from_hand(player, h, b)
            consumes_budget = True
        else:
            raise ValueError(f"Unknown action {action_int}")

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

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _submit_order(player: PlayerState) -> None:
        player.phase = PlayerPhase.DONE

    def _after_player_finished(self, state: MiniBGState, idx: int) -> None:
        minibg_round.after_player_finished(
            state,
            idx,
            fire_on_turn_end=self._shop_triggers.fire_on_turn_end,
            resolve_battle_and_advance=self._resolve_battle_and_advance,
        )

    # ------------------------------------------------------------------
    # Shop primitives
    # ------------------------------------------------------------------

    def _do_buy(self, player: PlayerState, slot: int) -> None:
        recruitment_economy.buy_from_shop(
            player,
            slot,
            on_bought=self._shop_triggers.fire_on_buy,
            on_triples=recruitment_triples.resolve_triples_loop,
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

    def _do_sell(self, player: PlayerState, pos: int) -> None:
        recruitment_economy.sell_from_board(
            player, pos, on_triples=recruitment_triples.resolve_triples_loop
        )

    def _do_roll(
        self, player: PlayerState, shop_excluded_race: Optional[Race]
    ) -> None:
        recruitment_economy.roll_shop(
            player, shop_excluded_race, rng=self._rng
        )

    def _do_level_up(
        self, player: PlayerState, shop_excluded_race: Optional[Race]
    ) -> None:
        recruitment_economy.level_up_tavern(
            player, shop_excluded_race, rng=self._rng
        )

    def _refresh_shop(
        self, player: PlayerState, shop_excluded_race: Optional[Race]
    ) -> None:
        recruitment_shop.refresh_shop(
            player, shop_excluded_race, rng=self._rng
        )

    def _refresh_shop_fill_empty_slots(
        self, player: PlayerState, shop_excluded_race: Optional[Race]
    ) -> None:
        recruitment_shop.refresh_shop_fill_empty_slots(
            player, shop_excluded_race, rng=self._rng
        )

    # ------------------------------------------------------------------
    # Round resolution

    def _resolve_battle_and_advance(self, state: MiniBGState) -> None:
        minibg_round.resolve_battle_and_advance(
            state,
            rng=self._rng,
            combat_board_max=COMBAT_BOARD_MAX,
            damage_cap=DAMAGE_CAP,
            board_size=BOARD_SIZE,
            max_rounds=MAX_ROUNDS,
            max_tier=MAX_TIER,
            level_up_discount_per_round=LEVEL_UP_DISCOUNT_PER_ROUND,
            fire_on_turn_start=self._shop_triggers.fire_on_turn_start,
            refresh_shop=self._refresh_shop,
            refresh_shop_fill_empty_slots=self._refresh_shop_fill_empty_slots,
        )

    def _fresh_player(
        self, round_number: int, shop_excluded_race: Optional[Race]
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
        self._refresh_shop(player, shop_excluded_race)
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
