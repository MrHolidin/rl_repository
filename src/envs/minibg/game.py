from __future__ import annotations

from copy import copy
from typing import List, Optional, Sequence

import numpy as np

from src.games.turn_based_game import Action as ActionType
from src.games.turn_based_game import TurnBasedGame

from .actions import (
    BOARD_SIZE,
    BUY_COST,
    HAND_SIZE,
    LEVEL_UP_COSTS,
    MAX_ROUNDS,
    MAX_SHOP_ACTIONS,
    MAX_TIER,
    ROLL_COST,
    SELL_REWARD,
    SHOP_SIZE,
    STARTING_HEALTH,
    STARTING_TIER,
    Action,
    gold_for_round,
)
from .battle import simulate_battle
from .cards import make_minion, shop_pool_for_tier
from .effects import BuffRandomFriendly, Trigger
from .state import MiniBGState, Minion, PlayerPhase, PlayerState


PLAYER_TOKENS = (1, -1)


class MiniBGGame(TurnBasedGame[MiniBGState]):
    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def initial_state(self) -> MiniBGState:
        initiative_player = int(self._rng.integers(0, 2))
        players = (
            self._fresh_player(round_number=1),
            self._fresh_player(round_number=1),
        )
        return MiniBGState(
            players=players,
            round_number=1,
            current_player_index=0,
            initiative_player=initiative_player,
            winner=None,
            done=False,
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

        actions: List[int] = []

        if player.phase == PlayerPhase.ORDER:
            # Order phase: a permutation must be submitted. The reorder is
            # produced by the env layer (`reorder_board(...)`) and immediately
            # followed by ``apply_action(FINISH)``; the only "game-level"
            # action emitted from the order phase is ``FINISH``.
            actions.append(int(Action.FINISH))
            return actions

        # Shop phase.
        can_act = player.shop_actions_used < MAX_SHOP_ACTIONS
        hand_full = sum(1 for m in player.hand if m is not None) >= HAND_SIZE
        board_full = len(player.board) >= BOARD_SIZE

        if can_act:
            if not hand_full:
                for slot in range(SHOP_SIZE):
                    if (
                        player.shop[slot] is not None
                        and player.gold >= BUY_COST
                    ):
                        actions.append(int(Action.BUY_SLOT_0) + slot)

            for pos in range(BOARD_SIZE):
                if pos < len(player.board):
                    actions.append(int(Action.SELL_BOARD_0) + pos)

            if not board_full:
                for h in range(HAND_SIZE):
                    if player.hand[h] is not None:
                        actions.append(int(Action.PLACE_HAND_0) + h)

            if player.gold >= ROLL_COST:
                actions.append(int(Action.ROLL))

            if (
                player.tavern_tier < MAX_TIER
                and player.gold >= LEVEL_UP_COSTS[player.tavern_tier]
            ):
                actions.append(int(Action.LEVEL_UP))

        actions.append(int(Action.FINISH))
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

        if player.phase == PlayerPhase.ORDER:
            # FINISH from order phase finalizes the round for this player.
            assert action_int == int(Action.FINISH)
            self._submit_order(player)
            self._after_player_finished(new_state, idx)
            return new_state

        # Shop phase.
        if action_int == int(Action.FINISH):
            # Phase flip only — same player keeps the turn to submit order.
            self._enter_order_phase(player)
            return new_state

        # Shop actions cost a budget slot.
        if Action.BUY_SLOT_0 <= action_int <= Action.BUY_SLOT_2:
            self._do_buy(player, action_int - int(Action.BUY_SLOT_0))
        elif Action.SELL_BOARD_0 <= action_int <= Action.SELL_BOARD_3:
            self._do_sell(player, action_int - int(Action.SELL_BOARD_0))
        elif action_int == int(Action.ROLL):
            self._do_roll(player)
        elif action_int == int(Action.LEVEL_UP):
            self._do_level_up(player)
        elif Action.PLACE_HAND_0 <= action_int <= Action.PLACE_HAND_2:
            self._do_place(player, action_int - int(Action.PLACE_HAND_0))
        else:
            raise ValueError(f"Unknown action {action_int}")

        player.shop_actions_used += 1
        if player.shop_actions_used >= MAX_SHOP_ACTIONS:
            # Budget exhausted -> auto-flip to order phase. Player still
            # owes a SELECT_ORDER on their next turn.
            self._enter_order_phase(player)

        return new_state

    def reorder_board(
        self,
        state: MiniBGState,
        player_idx: int,
        perm: Sequence[int],
    ) -> MiniBGState:
        if state.done:
            raise ValueError("Cannot reorder in terminal state")
        if len(perm) != BOARD_SIZE:
            raise ValueError(f"perm must have length {BOARD_SIZE}, got {len(perm)}")
        if sorted(perm) != list(range(BOARD_SIZE)):
            raise ValueError(
                f"perm must be a permutation of 0..{BOARD_SIZE - 1}: got {tuple(perm)}"
            )

        new_state = self._copy_state(state)
        player = new_state.players[player_idx]
        k = len(player.board)
        # Compact-after-permute: only old positions < k correspond to real
        # minions. Take their target positions in order, drop empties.
        # For k = BOARD_SIZE this is a normal reorder; for k < BOARD_SIZE
        # all 24 perms collapse into k! distinct outcomes.
        player.board = [player.board[p] for p in perm if p < k]
        return new_state

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _enter_order_phase(player: PlayerState) -> None:
        if player.phase == PlayerPhase.SHOP:
            player.phase = PlayerPhase.ORDER

    @staticmethod
    def _submit_order(player: PlayerState) -> None:
        player.phase = PlayerPhase.DONE

    def _after_player_finished(self, state: MiniBGState, idx: int) -> None:
        """Called after a player transitions to DONE."""
        self._fire_on_turn_end(state.players[idx])
        other_idx = 1 - idx
        if state.players[other_idx].phase != PlayerPhase.DONE:
            state.current_player_index = other_idx
        else:
            self._resolve_battle_and_advance(state)

    # ------------------------------------------------------------------
    # Shop primitives
    # ------------------------------------------------------------------

    def _do_buy(self, player: PlayerState, slot: int) -> None:
        minion = player.shop[slot]
        assert minion is not None
        player.gold -= BUY_COST
        player.shop[slot] = None
        # First empty hand slot.
        h = next(
            (i for i in range(HAND_SIZE) if player.hand[i] is None), None
        )
        assert h is not None, "BUY illegal when hand is full (legal mask bug)"
        player.hand[h] = minion
        self._fire_on_buy(minion, player)

    def _do_place(self, player: PlayerState, hand_slot: int) -> None:
        minion = player.hand[hand_slot]
        assert minion is not None
        assert len(player.board) < BOARD_SIZE
        player.hand[hand_slot] = None
        player.board.append(minion)

    def _do_sell(self, player: PlayerState, pos: int) -> None:
        del player.board[pos]
        player.gold += SELL_REWARD

    def _do_roll(self, player: PlayerState) -> None:
        player.gold -= ROLL_COST
        self._refresh_shop(player)

    def _do_level_up(self, player: PlayerState) -> None:
        cost = LEVEL_UP_COSTS[player.tavern_tier]
        player.gold -= cost
        player.tavern_tier += 1

    # ------------------------------------------------------------------
    # Effects
    # ------------------------------------------------------------------

    def _apply_buff_random(
        self,
        source: Minion,
        effect: BuffRandomFriendly,
        board: List[Minion],
    ) -> None:
        pool = (
            [m for m in board if m is not source]
            if effect.exclude_self
            else list(board)
        )
        if not pool:
            return
        target = pool[int(self._rng.integers(0, len(pool)))]
        target.bonus_attack += effect.attack
        target.bonus_health += effect.health

    def _fire_on_buy(self, minion: Minion, player: PlayerState) -> None:
        # ON_BUY targets the board only; hand cards are not buffable. The
        # bought minion sits in hand at trigger time, so it cannot be its own
        # target either. Empty board -> no-op.
        for ab in minion.abilities:
            if ab.trigger == Trigger.ON_BUY and isinstance(ab.effect, BuffRandomFriendly):
                self._apply_buff_random(minion, ab.effect, player.board)

    def _fire_on_turn_end(self, player: PlayerState) -> None:
        for source in list(player.board):
            for ab in source.abilities:
                if ab.trigger == Trigger.ON_TURN_END and isinstance(ab.effect, BuffRandomFriendly):
                    self._apply_buff_random(source, ab.effect, player.board)

    def _refresh_shop(self, player: PlayerState) -> None:
        pool = shop_pool_for_tier(player.tavern_tier)
        new_shop: List[Optional[Minion]] = []
        for _ in range(SHOP_SIZE):
            card_id = pool[int(self._rng.integers(0, len(pool)))]
            new_shop.append(make_minion(card_id))
        player.shop = new_shop

    # ------------------------------------------------------------------
    # Round resolution
    # ------------------------------------------------------------------

    def _resolve_battle_and_advance(self, state: MiniBGState) -> None:
        p0_has_initiative = (state.round_number % 2 == 1) == (state.initiative_player == 0)
        dmg_p0, dmg_p1 = simulate_battle(
            p0_board=state.players[0].board,
            p1_board=state.players[1].board,
            p0_has_initiative=p0_has_initiative,
            rng=self._rng,
        )
        state.players[0].health -= dmg_p0
        state.players[1].health -= dmg_p1

        p0_dead = state.players[0].health <= 0
        p1_dead = state.players[1].health <= 0

        if p0_dead or p1_dead:
            state.done = True
            if p0_dead and p1_dead:
                state.winner = 0
            elif p0_dead:
                state.winner = -1
            else:
                state.winner = 1
            return

        if state.round_number >= MAX_ROUNDS:
            state.done = True
            state.winner = 0
            return

        state.round_number += 1
        for p in state.players:
            p.gold = gold_for_round(state.round_number)
            p.phase = PlayerPhase.SHOP
            p.shop_actions_used = 0
            # Hand persists between rounds; only the shop is refreshed.
            self._refresh_shop(p)
        state.current_player_index = 0

    def _fresh_player(self, round_number: int) -> PlayerState:
        player = PlayerState(
            health=STARTING_HEALTH,
            gold=gold_for_round(round_number),
            tavern_tier=STARTING_TIER,
            board=[],
            shop=[None for _ in range(SHOP_SIZE)],
            hand=[None for _ in range(HAND_SIZE)],
            phase=PlayerPhase.SHOP,
            shop_actions_used=0,
        )
        self._refresh_shop(player)
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
        )

    @staticmethod
    def _copy_player(p: PlayerState) -> PlayerState:
        return PlayerState(
            health=p.health,
            gold=p.gold,
            tavern_tier=p.tavern_tier,
            board=[copy(m) for m in p.board],
            shop=[copy(m) if m is not None else None for m in p.shop],
            hand=[copy(m) if m is not None else None for m in p.hand],
            phase=p.phase,
            shop_actions_used=p.shop_actions_used,
        )


__all__ = ["MiniBGGame", "PLAYER_TOKENS"]
