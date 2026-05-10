from __future__ import annotations

from copy import copy
from typing import List, Optional, Sequence

import numpy as np

from src.games.turn_based_game import Action as ActionType
from src.games.turn_based_game import TurnBasedGame

from .actions import (
    BOARD_SIZE,
    BUY_COST,
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
from .state import MiniBGState, Minion, PlayerState


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
        if player.shopping_finished:
            return []

        actions: List[int] = []

        can_act = player.shop_actions_used < MAX_SHOP_ACTIONS
        board_full = len(player.board) >= BOARD_SIZE

        if can_act:
            for slot in range(SHOP_SIZE):
                if (
                    not board_full
                    and player.shop[slot] is not None
                    and player.gold >= BUY_COST
                ):
                    actions.append(int(Action.BUY_SLOT_0) + slot)

            for pos in range(BOARD_SIZE):
                if pos < len(player.board):
                    actions.append(int(Action.SELL_BOARD_0) + pos)

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
                f"gold={state.players[state.current_player_index].gold})"
            )

        new_state = self._copy_state(state)
        idx = new_state.current_player_index
        player = new_state.players[idx]

        if Action.BUY_SLOT_0 <= action_int <= Action.BUY_SLOT_2:
            slot = action_int - int(Action.BUY_SLOT_0)
            self._do_buy(player, slot)
        elif Action.SELL_BOARD_0 <= action_int <= Action.SELL_BOARD_3:
            pos = action_int - int(Action.SELL_BOARD_0)
            self._do_sell(player, pos)
        elif action_int == int(Action.ROLL):
            self._do_roll(player)
        elif action_int == int(Action.LEVEL_UP):
            self._do_level_up(player)
        elif action_int == int(Action.FINISH):
            player.shopping_finished = True
        else:
            raise ValueError(f"Unknown action {action_int}")

        if action_int != int(Action.FINISH):
            player.shop_actions_used += 1
            if player.shop_actions_used >= MAX_SHOP_ACTIONS:
                player.shopping_finished = True

        if player.shopping_finished:
            self._fire_on_turn_end(player)
            other_idx = 1 - idx
            if not new_state.players[other_idx].shopping_finished:
                new_state.current_player_index = other_idx
            else:
                self._resolve_battle_and_advance(new_state)

        return new_state

    def _do_buy(self, player: PlayerState, slot: int) -> None:
        minion = player.shop[slot]
        assert minion is not None
        player.gold -= BUY_COST
        player.shop[slot] = None
        player.board.append(minion)
        self._fire_on_buy(minion, player)

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
            p.shopping_finished = False
            p.shop_actions_used = 0
            self._refresh_shop(p)
        state.current_player_index = 0

    def _fresh_player(self, round_number: int) -> PlayerState:
        player = PlayerState(
            health=STARTING_HEALTH,
            gold=gold_for_round(round_number),
            tavern_tier=STARTING_TIER,
            board=[],
            shop=[None, None, None],
            shopping_finished=False,
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
            shopping_finished=p.shopping_finished,
            shop_actions_used=p.shop_actions_used,
        )


__all__ = ["MiniBGGame", "PLAYER_TOKENS"]
