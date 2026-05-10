from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from ..action_map import (
    A_BUY_BASE,
    A_FINISH,
    A_LEVEL_UP,
    A_PLACE_BASE,
    A_ROLL,
    A_SELECT_ORDER_BASE,
    A_SELL_BASE,
    NUM_ENV_ACTIONS,
)
from ..actions import BUY_COST, HAND_SIZE, MAX_TIER, SHOP_SIZE
from ..env import MiniBGEnv
from ..state import Minion, PlayerState

from ..battle import simulate_battle
from ..effects import Keyword, Trigger
from .common import (
    BASE_PRIORITY,
    BUFFER_T2_GOOD_TARGET_IDS,
    TOKEN_BUFFER_OK_IDS,
    best_buy_slot,
    board_full,
    board_value,
    buffer_is_good,
    can_afford_level,
    choose_final_order,
    has_good_buffer_target,
    has_other_minion,
    incoming_shop_minion,
    keyword_effect_bonus,
    legal_env_indices,
    order_key_default,
    order_key_token,
    reset_shop_counters_if_needed,
    score_minion_on_board,
    shop_minions_with_slots,
    worst_board_sell_index,
)


def _mask(env: MiniBGEnv) -> np.ndarray:
    return env.legal_actions_mask


def _me(env: MiniBGEnv) -> PlayerState:
    return env.state.players[env.current_player()]


def _has_legal_buy(mask: np.ndarray) -> bool:
    return any(bool(mask[A_BUY_BASE + s]) for s in range(SHOP_SIZE))


class HeuristicBot(ABC):
    name: str = "heuristic"
    order_style: str = "default"

    def __init__(self) -> None:
        self._rolls: List[int] = [0]
        self._last_shop_used: List[int] = [-1]

    def _sync(self, p: PlayerState) -> None:
        reset_shop_counters_if_needed(p, self._rolls, self._last_shop_used)

    def _finish(self, env: MiniBGEnv) -> int:
        """Default end-of-decision behaviour.

        In the order phase only SELECT_ORDER_* are legal -> pick a permutation.
        In the shop phase prefer placing pending hand cards before issuing the
        explicit FINISH (no point in saving cards for next round in heuristic
        play). Bots that want to defer hand cards should override this.
        """
        mask = _mask(env)
        p = _me(env)

        if not bool(mask[A_FINISH]):
            key = order_key_token if self.order_style == "token" else order_key_default
            return choose_final_order(p.board, mask, key)

        for i in range(HAND_SIZE):
            if bool(mask[A_PLACE_BASE + i]):
                return A_PLACE_BASE + i

        return A_FINISH

    @abstractmethod
    def choose_action(self, env: MiniBGEnv) -> int:
        raise NotImplementedError


class RandomBot(HeuristicBot):
    name = "random"
    order_style = "default"

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__()
        self._rng = np.random.default_rng(seed)

    def choose_action(self, env: MiniBGEnv) -> int:
        legal = legal_env_indices(_mask(env))
        return int(self._rng.choice(legal))


class TempoBot(HeuristicBot):
    name = "tempo"
    order_style = "default"

    def _tempo_score(self, m: Minion, p: PlayerState) -> float:
        cid = m.card_id
        s = float(BASE_PRIORITY.get(cid, 5))
        if cid == "buffer" and not buffer_is_good(p.board):
            s -= 25.0
        return s

    def _want_level(self, env: MiniBGEnv, p: PlayerState) -> bool:
        r = env.state.round_number
        if p.tavern_tier >= MAX_TIER:
            return False
        if p.tavern_tier == 1 and r < 4:
            return False
        if p.tavern_tier == 2 and r < 7:
            return False
        return True

    def choose_action(self, env: MiniBGEnv) -> int:
        mask = _mask(env)
        p = _me(env)
        self._sync(p)

        if bool(mask[A_LEVEL_UP]) and self._want_level(env, p) and can_afford_level(p):
            return A_LEVEL_UP

        slot = best_buy_slot(p, mask, self._tempo_score)
        if slot is not None and not board_full(p):
            return A_BUY_BASE + slot

        if slot is not None and board_full(p):
            incoming = incoming_shop_minion(p, slot)
            if incoming is not None:
                worst = worst_board_sell_index(p)
                if worst is not None:
                    my_worst = self._tempo_score(p.board[worst], p)
                    inc = self._tempo_score(incoming, p)
                    if inc > my_worst + 6.0 and bool(mask[A_SELL_BASE + worst]):
                        return A_SELL_BASE + worst

        if (
            bool(mask[A_ROLL])
            and p.gold >= 4
            and not _has_legal_buy(mask)
            and self._rolls[0] < 1
        ):
            self._rolls[0] += 1
            return A_ROLL

        return self._finish(env)


class BufferT1Bot(HeuristicBot):
    name = "buffer_t1"
    order_style = "default"

    def _score(self, m: Minion, p: PlayerState) -> float:
        good = buffer_is_good(p.board)
        cid = m.card_id
        if cid == "buffer":
            return 30.0 if good else -50.0
        if cid == "guard":
            return 15.0
        if cid == "recruit":
            return 12.0
        return float(BASE_PRIORITY.get(cid, 5))

    def choose_action(self, env: MiniBGEnv) -> int:
        mask = _mask(env)
        p = _me(env)
        self._sync(p)

        if len(p.board) == 0:
            for cid_pref in ("guard", "recruit"):
                for s, m in shop_minions_with_slots(p.shop):
                    if m.card_id == cid_pref and bool(mask[A_BUY_BASE + s]):
                        return A_BUY_BASE + s
            for s, m in shop_minions_with_slots(p.shop):
                if m.card_id != "buffer" and bool(mask[A_BUY_BASE + s]):
                    return A_BUY_BASE + s

        if buffer_is_good(p.board):
            for s, m in shop_minions_with_slots(p.shop):
                if m.card_id == "buffer" and bool(mask[A_BUY_BASE + s]):
                    if not board_full(p):
                        return A_BUY_BASE + s
                    w = worst_board_sell_index(p)
                    if w is not None and p.board[w].card_id in ("recruit", "buffer"):
                        if bool(mask[A_SELL_BASE + w]):
                            return A_SELL_BASE + w

        slot = best_buy_slot(p, mask, self._score)
        if slot is not None and not board_full(p):
            return A_BUY_BASE + slot

        if (
            bool(mask[A_ROLL])
            and p.gold >= 1
            and buffer_is_good(p.board)
            and not any(m.card_id == "buffer" for _, m in shop_minions_with_slots(p.shop))
            and self._rolls[0] < 6
        ):
            self._rolls[0] += 1
            return A_ROLL

        if slot is not None and board_full(p):
            w = worst_board_sell_index(p)
            if w is not None and bool(mask[A_SELL_BASE + w]):
                return A_SELL_BASE + w

        return self._finish(env)


class WideT1Bot(HeuristicBot):
    name = "wide_t1"
    order_style = "default"

    def _t1_only(self, m: Minion) -> bool:
        return m.tier == 1

    def _score(self, m: Minion, p: PlayerState) -> float:
        if m.tier != 1:
            return -1000.0
        if m.card_id == "guard":
            return 20.0
        if m.card_id == "recruit":
            return 15.0
        if m.card_id == "buffer":
            return 12.0 if buffer_is_good(p.board) else -20.0
        return 5.0

    def choose_action(self, env: MiniBGEnv) -> int:
        mask = _mask(env)
        p = _me(env)
        self._sync(p)

        if not board_full(p):
            slot = best_buy_slot(p, mask, self._score)
            if slot is not None:
                m = incoming_shop_minion(p, slot)
                if m is not None and self._t1_only(m):
                    return A_BUY_BASE + slot

        if board_full(p):
            for s, m in shop_minions_with_slots(p.shop):
                if m.card_id == "buffer" and buffer_is_good(p.board):
                    if bool(mask[A_BUY_BASE + s]):
                        return A_BUY_BASE + s
                    w = worst_board_sell_index(p)
                    if w is not None and p.board[w].card_id in ("buffer", "recruit"):
                        if bool(mask[A_SELL_BASE + w]):
                            return A_SELL_BASE + w

        if (
            bool(mask[A_ROLL])
            and p.gold >= 4
            and not _has_legal_buy(mask)
            and self._rolls[0] < 1
        ):
            self._rolls[0] += 1
            return A_ROLL

        return self._finish(env)


class EarlyT2PressureBot(HeuristicBot):
    name = "early_t2_pressure"
    order_style = "default"

    def _t2_score(self, m: Minion, p: PlayerState) -> float:
        cid = m.card_id
        if m.tier == 2:
            pri = {
                "shield_bot": 35.0,
                "pack_rat": 33.0,
                "bruiser": 31.0,
            }.get(cid, 18.0)
        else:
            pri = {"guard": 16.0, "recruit": 12.0, "buffer": 10.0}.get(cid, 5.0)
        if cid == "buffer" and buffer_is_good(p.board):
            pri += 12.0
        if cid == "buffer" and not buffer_is_good(p.board):
            pri -= 15.0
        return pri

    def choose_action(self, env: MiniBGEnv) -> int:
        mask = _mask(env)
        p = _me(env)
        self._sync(p)
        r = env.state.round_number

        if r == 2 and bool(mask[A_LEVEL_UP]) and can_afford_level(p):
            return A_LEVEL_UP

        if bool(mask[A_LEVEL_UP]) and p.tavern_tier == 2 and r >= 8 and p.gold >= 6:
            return A_LEVEL_UP

        slot = best_buy_slot(p, mask, self._t2_score)
        if slot is not None and not board_full(p):
            return A_BUY_BASE + slot

        if slot is not None and board_full(p):
            inc = incoming_shop_minion(p, slot)
            if inc is not None and self._t2_score(inc, p) >= 28.0:
                w = worst_board_sell_index(p)
                if w is not None and bool(mask[A_SELL_BASE + w]):
                    return A_SELL_BASE + w

        if (
            bool(mask[A_ROLL])
            and p.tavern_tier >= 2
            and p.gold >= 1
            and not _has_legal_buy(mask)
            and self._rolls[0] < 3
        ):
            self._rolls[0] += 1
            return A_ROLL

        return self._finish(env)


class DelayedT2PressureBot(HeuristicBot):
    name = "delayed_t2_pressure"
    order_style = "default"

    def _score(self, m: Minion, p: PlayerState) -> float:
        cid = m.card_id
        if m.tier == 2:
            return float(
                {"shield_bot": 35, "pack_rat": 33, "bruiser": 31}.get(cid, 18)
            )
        if len(p.board) >= 2 and cid == "buffer":
            return 18.0
        if cid == "buffer" and not buffer_is_good(p.board):
            return -10.0
        return float(BASE_PRIORITY.get(cid, 8))

    def choose_action(self, env: MiniBGEnv) -> int:
        mask = _mask(env)
        p = _me(env)
        self._sync(p)
        r = env.state.round_number

        if r == 3 and bool(mask[A_LEVEL_UP]) and can_afford_level(p):
            return A_LEVEL_UP

        if bool(mask[A_LEVEL_UP]) and p.tavern_tier == 2 and r >= 7 and can_afford_level(p):
            return A_LEVEL_UP

        slot = best_buy_slot(p, mask, self._score)
        if slot is not None and not board_full(p):
            return A_BUY_BASE + slot

        if slot is not None and board_full(p):
            w = worst_board_sell_index(p)
            if w is not None and bool(mask[A_SELL_BASE + w]):
                return A_SELL_BASE + w

        if bool(mask[A_ROLL]) and p.tavern_tier >= 2 and self._rolls[0] < 3 and p.gold >= 1:
            if not _has_legal_buy(mask):
                self._rolls[0] += 1
                return A_ROLL

        return self._finish(env)


class BalancedBot(HeuristicBot):
    name = "balanced"
    order_style = "default"

    def _score(self, m: Minion, p: PlayerState) -> float:
        cid = m.card_id
        s = float(BASE_PRIORITY.get(cid, 8))
        if cid == "buffer" and not buffer_is_good(p.board):
            s -= 20.0
        if cid == "mentor":
            if p.tavern_tier >= 3 and has_other_minion(p.board):
                s += 8.0
            else:
                s -= 10.0
        return s

    def choose_action(self, env: MiniBGEnv) -> int:
        mask = _mask(env)
        p = _me(env)
        self._sync(p)
        r = env.state.round_number

        if r == 3 and bool(mask[A_LEVEL_UP]) and can_afford_level(p):
            return A_LEVEL_UP

        if (
            r == 5
            and p.tavern_tier == 2
            and p.health >= 7
            and bool(mask[A_LEVEL_UP])
            and can_afford_level(p)
        ):
            return A_LEVEL_UP

        if p.health > 6 and bool(mask[A_LEVEL_UP]) and can_afford_level(p):
            if p.tavern_tier == 1 and r >= 4:
                return A_LEVEL_UP

        slot = best_buy_slot(p, mask, self._score)
        if slot is not None and not board_full(p):
            return A_BUY_BASE + slot

        if p.health <= 6:
            if bool(mask[A_ROLL]) and p.gold >= 1 and self._rolls[0] < 4:
                if not _has_legal_buy(mask):
                    self._rolls[0] += 1
                    return A_ROLL

        if slot is not None and board_full(p):
            w = worst_board_sell_index(p)
            if w is not None and bool(mask[A_SELL_BASE + w]):
                return A_SELL_BASE + w

        if bool(mask[A_ROLL]) and p.gold >= 2 and self._rolls[0] < 2 and not _has_legal_buy(mask):
            self._rolls[0] += 1
            return A_ROLL

        return self._finish(env)


class TokenBot(HeuristicBot):
    name = "token"
    order_style = "token"

    def _pri(self, m: Minion) -> float:
        return float(
            {
                "pack_rat": 40,
                "summoner": 39,
                "commander": 38,
                "guard": 28,
                "shield_bot": 26,
                "mentor": 24,
                "big_guy": 22,
                "bruiser": 18,
                "buffer": 10,
                "recruit": 8,
            }.get(m.card_id, 5)
        )

    def _score(self, m: Minion, p: PlayerState) -> float:
        s = self._pri(m)
        if m.card_id == "buffer":
            if any(x.card_id in TOKEN_BUFFER_OK_IDS for x in p.board):
                s += 18.0
            else:
                s -= 12.0
        return s

    def choose_action(self, env: MiniBGEnv) -> int:
        mask = _mask(env)
        p = _me(env)
        self._sync(p)
        r = env.state.round_number

        if p.tavern_tier == 1 and r in (2, 3) and bool(mask[A_LEVEL_UP]) and can_afford_level(p):
            return A_LEVEL_UP

        if p.tavern_tier == 2 and r in (4, 5) and bool(mask[A_LEVEL_UP]) and can_afford_level(p):
            return A_LEVEL_UP

        for s, m in shop_minions_with_slots(p.shop):
            if m.card_id == "pack_rat" and bool(mask[A_BUY_BASE + s]):
                if not board_full(p):
                    return A_BUY_BASE + s
                w = worst_board_sell_index(p)
                if w is not None and p.board[w].card_id in ("recruit", "buffer", "bruiser"):
                    if bool(mask[A_SELL_BASE + w]):
                        return A_SELL_BASE + w

        slot = best_buy_slot(p, mask, self._score)
        if slot is not None and not board_full(p):
            return A_BUY_BASE + slot

        if board_full(p):
            for s, m in shop_minions_with_slots(p.shop):
                if self._score(m, p) >= 30.0:
                    w = worst_board_sell_index(p)
                    if w is not None and p.board[w].card_id in ("recruit", "buffer", "bruiser"):
                        if bool(mask[A_SELL_BASE + w]):
                            return A_SELL_BASE + w

        if bool(mask[A_ROLL]) and p.gold >= 1 and self._rolls[0] < 4:
            if not _has_legal_buy(mask):
                self._rolls[0] += 1
                return A_ROLL

        return self._finish(env)


class FastT3Bot(HeuristicBot):
    name = "fast_t3"
    order_style = "default"

    def _score(self, m: Minion, p: PlayerState) -> float:
        tier = p.tavern_tier
        if tier >= 3 and m.tier == 3:
            return float(
                {"big_guy": 36, "summoner": 35, "commander": 34, "mentor": 33}.get(
                    m.card_id, 20
                )
            )
        if tier == 2:
            return float(
                {"shield_bot": 30, "pack_rat": 29, "bruiser": 28}.get(m.card_id, 12)
            )
        return float(BASE_PRIORITY.get(m.card_id, 8))

    def choose_action(self, env: MiniBGEnv) -> int:
        mask = _mask(env)
        p = _me(env)
        self._sync(p)
        r = env.state.round_number

        if r == 2 and bool(mask[A_LEVEL_UP]) and can_afford_level(p):
            return A_LEVEL_UP

        if r == 4 and p.tavern_tier == 2 and bool(mask[A_LEVEL_UP]) and can_afford_level(p):
            return A_LEVEL_UP

        slot = best_buy_slot(p, mask, self._score)
        if slot is not None and not board_full(p):
            return A_BUY_BASE + slot

        if p.tavern_tier >= 3 and board_full(p) and slot is not None:
            inc = incoming_shop_minion(p, slot)
            if inc is not None and inc.tier == 3:
                w = worst_board_sell_index(p)
                if w is not None and p.board[w].tier == 1 and bool(mask[A_SELL_BASE + w]):
                    return A_SELL_BASE + w

        if p.tavern_tier >= 3 and bool(mask[A_ROLL]) and not _has_legal_buy(mask):
            if self._rolls[0] < 4 and p.gold >= 1:
                self._rolls[0] += 1
                return A_ROLL

        if bool(mask[A_ROLL]) and p.gold >= 1 and self._rolls[0] < 2 and not _has_legal_buy(mask):
            self._rolls[0] += 1
            return A_ROLL

        return self._finish(env)


class T3ScalerBot(HeuristicBot):
    name = "t3_scaler"
    order_style = "default"

    def _score(self, m: Minion, p: PlayerState) -> float:
        cid = m.card_id
        if p.tavern_tier >= 3:
            base = {
                "mentor": 50.0,
                "big_guy": 35.0,
                "summoner": 34.0,
                "commander": 32.0,
                "shield_bot": 25.0,
                "pack_rat": 24.0,
                "bruiser": 22.0,
                "buffer": 20.0,
                "guard": 15.0,
                "recruit": 8.0,
            }.get(cid, 10.0)
            if cid == "mentor" and not has_other_minion(p.board):
                base -= 30.0
            return base
        return float(BASE_PRIORITY.get(cid, 8))

    def choose_action(self, env: MiniBGEnv) -> int:
        mask = _mask(env)
        p = _me(env)
        self._sync(p)
        r = env.state.round_number

        if r == 2 and bool(mask[A_LEVEL_UP]) and can_afford_level(p):
            return A_LEVEL_UP

        if r == 4 and p.tavern_tier == 2 and bool(mask[A_LEVEL_UP]) and can_afford_level(p):
            return A_LEVEL_UP

        for s, m in shop_minions_with_slots(p.shop):
            if m.card_id == "mentor" and buffer_is_good(p.board) and bool(mask[A_BUY_BASE + s]):
                if not board_full(p):
                    return A_BUY_BASE + s
                w = worst_board_sell_index(p)
                if w is not None and bool(mask[A_SELL_BASE + w]):
                    return A_SELL_BASE + w

        slot = best_buy_slot(p, mask, self._score)
        if slot is not None and not board_full(p):
            return A_BUY_BASE + slot

        if slot is not None and board_full(p):
            inc = incoming_shop_minion(p, slot)
            if inc is not None and inc.card_id == "mentor":
                w = worst_board_sell_index(p)
                if w is not None and bool(mask[A_SELL_BASE + w]):
                    return A_SELL_BASE + w

        if bool(mask[A_ROLL]) and p.tavern_tier >= 3 and p.gold >= 1 and self._rolls[0] < 5:
            if not _has_legal_buy(mask):
                self._rolls[0] += 1
                return A_ROLL

        return self._finish(env)


class BufferT2Bot(HeuristicBot):
    name = "buffer_t2"
    order_style = "default"

    def _has_good_target(self, p: PlayerState) -> bool:
        return any(m.card_id in BUFFER_T2_GOOD_TARGET_IDS for m in p.board)

    def _is_core(self, m: Minion) -> bool:
        if m.card_id not in ("shield_bot", "guard", "pack_rat", "bruiser"):
            return False
        return m.bonus_attack + m.bonus_health >= 2

    def _score(self, m: Minion, p: PlayerState) -> float:
        gt = self._has_good_target(p)
        if m.card_id == "buffer":
            return 45.0 if gt else 8.0
        return float(
            {
                "shield_bot": 35.0,
                "pack_rat": 33.0,
                "bruiser": 31.0,
                "guard": 25.0,
                "recruit": 10.0,
            }.get(m.card_id, 8.0)
        )

    def choose_action(self, env: MiniBGEnv) -> int:
        mask = _mask(env)
        p = _me(env)
        self._sync(p)
        r = env.state.round_number

        if r == 2 and bool(mask[A_LEVEL_UP]) and can_afford_level(p):
            return A_LEVEL_UP

        if board_full(p):
            for s, m in shop_minions_with_slots(p.shop):
                if m.card_id == "buffer" and self._has_good_target(p):
                    w = next(
                        (
                            i
                            for i, bm in enumerate(p.board)
                            if not self._is_core(bm)
                        ),
                        None,
                    )
                    if w is not None and bool(mask[A_SELL_BASE + w]):
                        return A_SELL_BASE + w

        slot = best_buy_slot(p, mask, self._score)
        if slot is not None and not board_full(p):
            return A_BUY_BASE + slot

        if bool(mask[A_LEVEL_UP]) and p.tavern_tier == 2 and r >= 11 and p.gold >= 8:
            return A_LEVEL_UP

        if bool(mask[A_ROLL]) and p.tavern_tier == 2 and self._rolls[0] < 3 and p.gold >= 1:
            if not _has_legal_buy(mask):
                self._rolls[0] += 1
                return A_ROLL

        return self._finish(env)


class BufferT2T3Bot(HeuristicBot):
    name = "buffer_t2_t3"
    order_style = "default"

    def _stable(self, p: PlayerState) -> bool:
        return board_value(p.board) >= 28 or p.health >= 9

    def _score_t2(self, m: Minion, p: PlayerState) -> float:
        s = float(BASE_PRIORITY.get(m.card_id, 8))
        if m.card_id == "buffer" and buffer_is_good(p.board):
            s += 25.0
        return s

    def _score_t3(self, m: Minion, p: PlayerState) -> float:
        return float(
            {
                "mentor": 42.0,
                "big_guy": 38.0,
                "summoner": 36.0,
                "commander": 34.0,
                "buffer": 28.0 if buffer_is_good(p.board) else 6.0,
            }.get(m.card_id, 12.0)
        )

    def choose_action(self, env: MiniBGEnv) -> int:
        mask = _mask(env)
        p = _me(env)
        self._sync(p)
        r = env.state.round_number

        if r == 2 and bool(mask[A_LEVEL_UP]) and can_afford_level(p):
            return A_LEVEL_UP

        if p.tavern_tier == 2 and r >= 5 and self._stable(p):
            if bool(mask[A_LEVEL_UP]) and can_afford_level(p):
                return A_LEVEL_UP

        score_fn = self._score_t3 if p.tavern_tier >= 3 else self._score_t2
        slot = best_buy_slot(p, mask, score_fn)
        if slot is not None and not board_full(p):
            return A_BUY_BASE + slot

        if slot is not None and board_full(p):
            w = worst_board_sell_index(p)
            if w is not None and bool(mask[A_SELL_BASE + w]):
                return A_SELL_BASE + w

        if bool(mask[A_ROLL]) and p.gold >= 1 and self._rolls[0] < 3:
            if not _has_legal_buy(mask):
                self._rolls[0] += 1
                return A_ROLL

        return self._finish(env)


class T2IntoT3PressureBot(HeuristicBot):
    name = "t2_into_t3_pressure"
    order_style = "default"

    def _weak(self, p: PlayerState) -> bool:
        return board_value(p.board) < 20

    def _score(self, m: Minion, p: PlayerState) -> float:
        tier = p.tavern_tier
        cid = m.card_id
        if tier >= 3:
            return float(
                {
                    "mentor": 40.0,
                    "big_guy": 38.0,
                    "summoner": 37.0,
                    "commander": 34.0,
                    "buffer": 30.0 if has_good_buffer_target(p.board) else 10.0,
                    "shield_bot": 26.0,
                    "pack_rat": 25.0,
                }.get(cid, 12.0)
            )
        pri = {
            "shield_bot": 35.0,
            "pack_rat": 34.0,
            "bruiser": 32.0,
            "guard": 20.0,
            "recruit": 12.0,
        }.get(cid, 8.0)
        if cid == "buffer":
            pri = 28.0 if has_good_buffer_target(p.board) else 10.0
        return pri

    def choose_action(self, env: MiniBGEnv) -> int:
        mask = _mask(env)
        p = _me(env)
        self._sync(p)
        r = env.state.round_number

        if r == 2 and bool(mask[A_LEVEL_UP]) and can_afford_level(p):
            return A_LEVEL_UP

        if r == 4 and p.tavern_tier == 2:
            if not self._weak(p) and bool(mask[A_LEVEL_UP]) and can_afford_level(p):
                return A_LEVEL_UP

        if r >= 5 and p.tavern_tier == 2 and bool(mask[A_LEVEL_UP]) and can_afford_level(p):
            return A_LEVEL_UP

        slot = best_buy_slot(p, mask, self._score)
        if slot is not None and not board_full(p):
            return A_BUY_BASE + slot

        if board_full(p) and slot is not None:
            inc = incoming_shop_minion(p, slot)
            if inc is not None:
                w = worst_board_sell_index(p)
                if w is not None and p.board[w].tier == 1 and bool(mask[A_SELL_BASE + w]):
                    if inc.tier >= 2 or inc.card_id in ("buffer", "mentor"):
                        return A_SELL_BASE + w

        if bool(mask[A_ROLL]) and p.gold >= 1 and self._rolls[0] < 4:
            if p.tavern_tier == 2 and self._weak(p):
                self._rolls[0] += 1
                return A_ROLL
            if p.tavern_tier >= 3 and not _has_legal_buy(mask):
                self._rolls[0] += 1
                return A_ROLL

        return self._finish(env)


_STRONG_TARGET_IDS = frozenset(
    {"shield_bot", "pack_rat", "bruiser", "guard", "big_guy",
     "commander", "summoner", "mentor"}
)


def _strong_target_count(board: List[Minion]) -> int:
    return sum(1 for m in board if m.card_id in _STRONG_TARGET_IDS)


def _board_min_count_for_initiative(board: List[Minion]) -> int:
    return len(board)


class ApexBot(HeuristicBot):
    """Handcrafted strong heuristic.

    Plan:
      R1: buy best T1 (avoid lone buffer).
      R2: level T1->T2 (always).
      R3: T2; buy best T2 (shield_bot / pack_rat / bruiser).
      R4: T2->T3 levelup if board has >=2 non-buffer cores AND we are not
           critically low on HP (<=10) nor far behind on life total (>=7 vs
           opponent); otherwise stay T2 one more shop to stabilize tempo.
      R5+: T3; mentor priority (snowball), big_guy / summoner / commander;
           refill via sell-worst when full and a clearly better minion
           is in shop; roll aggressively at T3 for mentor / big T3s.
    """

    name = "apex"
    order_style = "default"
    # HP / roll / R4-level gates in early game (set False for A/B baseline).
    early_hp_tempo: bool = True

    def _round(self, env: MiniBGEnv) -> int:
        return env.state.round_number

    def _strong_targets(self, p: PlayerState) -> int:
        return _strong_target_count(p.board)

    def _mentor_buy_priority_ok(self, p: PlayerState, r: int) -> bool:
        """Whether to run the early-shop mentor buy / sell-for-mentor block."""
        return True

    def _opponent_state(self, env: MiniBGEnv) -> PlayerState:
        idx = env.state.current_player_index
        return env.state.players[1 - idx]

    def _hp_gap_favors_opponent(self, env: MiniBGEnv, p: PlayerState) -> int:
        """Opponent HP minus ours (positive => we are behind on life)."""
        return int(self._opponent_state(env).health - p.health)

    def _early_hp_tempo_bonus(self, m: Minion, p: PlayerState, r: int) -> float:
        """Prefer bodies / shield / taunt in early shop when behind on HP."""
        if not self.early_hp_tempo:
            return 0.0
        env = getattr(self, "_shop_context_env", None)
        if env is None or r > 5 or p.tavern_tier >= 3:
            return 0.0
        gap = self._hp_gap_favors_opponent(env, p)
        if gap < 3:
            return 0.0
        b = 0.0
        if m.tier >= 2:
            b += min(5.0, 1.0 + gap * 0.35)
        if Keyword.SHIELD in m.keywords:
            b += min(5.0, 1.5 + gap * 0.3)
        if Keyword.TAUNT in m.keywords:
            b += min(3.0, 1.0 + gap * 0.2)
        if m.card_id in ("bruiser", "pack_rat", "guard", "shield_bot"):
            b += min(4.0, 1.2 + gap * 0.28)
        if m.card_id == "recruit" and len(p.board) < 4:
            b -= min(4.0, gap * 0.35)
        return b

    def _buy_score(self, m: Minion, p: PlayerState, r: int) -> float:
        """Score a candidate buy.

        Calibration: T1 minions ~ 8-10, T2 ~ 14-17, T3 ~ 22-30.
        Buffer is ALWAYS less valuable than a real T2/T3 minion in tempo
        terms; only meaningful on a stable, mostly-full board where the
        +1/+1 stacks on real cores.
        """
        cid = m.card_id
        tier = m.tier

        if cid == "buffer":
            non_buffer = sum(1 for x in p.board if x.card_id != "buffer")
            if non_buffer == 0:
                return -50.0
            tgt = self._strong_targets(p)
            # +1/+1 to a random other = expected ~2 stats added.
            # Buffer body itself = 1/1 = 2 stats.
            # Tier bonus small (T1 only).
            s = 4.0 + min(tgt, 3) * 1.2
            # Late-game with full board of cores -> buffer scaling shines
            if r >= 5 and tgt >= 2:
                s += 2.0
            if tgt == 0:
                s -= 3.0
            return s

        s = float(m.raw_attack + m.max_health) + tier * 4.0
        if Keyword.TAUNT in m.keywords:
            s += 1.0
        if Keyword.SHIELD in m.keywords:
            s += 4.0
        for ab in m.abilities:
            if ab.trigger == Trigger.ON_DEATH:
                s += 3.0
            elif ab.trigger == Trigger.AURA:
                s += 4.0 + min(3, len(p.board)) * 1.2
            elif ab.trigger == Trigger.ON_TURN_END:
                # Mentor: snowball if there is at least one strong target
                # and rounds left to compound.  +2/+1 per shop turn.
                rounds_left = max(0, 12 - r)
                tgt = self._strong_targets(p)
                if tgt >= 1:
                    s += 8.0 + rounds_left * 1.5
                else:
                    s -= 6.0
        if cid == "recruit":
            s -= 0.5
        s += self._early_hp_tempo_bonus(m, p, r)
        return s

    def _want_level(self, env: MiniBGEnv, p: PlayerState) -> bool:
        r = env.state.round_number
        if p.tavern_tier >= MAX_TIER:
            return False
        if not can_afford_level(p):
            return False
        if p.tavern_tier == 1:
            # Always level R2 (matches fast_t3 winning policy).
            return r >= 2
        if p.tavern_tier == 2:
            non_buffer = sum(1 for m in p.board if m.card_id != "buffer")
            # T2->T3 at R4 only if board can fight on its own (>= 2 cores).
            # Skipping R4 levelup for weak boards prevents the disastrous
            # 2-minion-vs-4 battle and lets us buy 2 T2 cores instead.
            if r == 4:
                if self.early_hp_tempo:
                    gap = self._hp_gap_favors_opponent(env, p)
                    if p.health <= 10 or gap >= 7:
                        return False
                return non_buffer >= 2
            # R5+: must level eventually to access T3 minions for late game.
            if r >= 5:
                return True
            return False
        return False

    def _best_sell_for_incoming(
        self, p: PlayerState, incoming: Minion, mask: np.ndarray,
        margin: float = 3.0,
    ) -> Optional[int]:
        if not p.board:
            return None
        # Sell-then-buy is only useful if we can actually afford the buy:
        # SELL refunds 1 gold, so we need gold >= BUY_COST - 1 = 2.
        if p.gold < BUY_COST - 1:
            return None
        my_scores = [(i, score_minion_on_board(m)) for i, m in enumerate(p.board)]
        incoming_score = score_minion_on_board(incoming)
        my_scores.sort(key=lambda t: t[1])
        worst_idx, worst_score = my_scores[0]
        # Incoming must clearly outclass the worst on board.
        if incoming_score - worst_score < margin:
            return None
        if not bool(mask[A_SELL_BASE + worst_idx]):
            return None
        return worst_idx

    def _find_best_in_shop(
        self,
        p: PlayerState,
        r: int,
    ) -> Tuple[Optional[int], Optional[Minion], float]:
        """Best shop slot ignoring BUY mask (so sell-then-buy can be planned)."""
        best_slot: Optional[int] = None
        best_minion: Optional[Minion] = None
        best_score = -1e18
        for slot, m in shop_minions_with_slots(p.shop):
            sc = self._buy_score(m, p, r)
            if sc > best_score:
                best_score = sc
                best_slot = slot
                best_minion = m
        return best_slot, best_minion, best_score

    def _max_rolls(self, p: PlayerState, r: int) -> int:
        if p.tavern_tier == 1:
            return 0
        if p.tavern_tier == 2:
            return 1
        # T3
        if r <= 4:
            return 1
        if r <= 6:
            return 3
        return 4

    def _shop_has_strong_t3(self, p: PlayerState) -> bool:
        return any(
            m.card_id in ("mentor", "big_guy", "summoner", "commander")
            for _, m in shop_minions_with_slots(p.shop)
        )

    def choose_action(self, env: MiniBGEnv) -> int:
        self._shop_context_env = env
        try:
            mask = _mask(env)
            p = _me(env)
            self._sync(p)
            r = env.state.round_number

            # --- Mentor priority: at T3, secure mentor if we have targets.
            if (
                p.tavern_tier >= 3
                and self._strong_targets(p) >= 1
                and self._mentor_buy_priority_ok(p, r)
            ):
                for s, m in shop_minions_with_slots(p.shop):
                    if m.card_id != "mentor":
                        continue
                    if not board_full(p) and bool(mask[A_BUY_BASE + s]):
                        return A_BUY_BASE + s
                    if board_full(p):
                        # Lower margin for mentor since it's a snowball card.
                        sell_idx = self._best_sell_for_incoming(p, m, mask, margin=2.0)
                        if sell_idx is not None:
                            return A_SELL_BASE + sell_idx

            # --- Level-up.
            if bool(mask[A_LEVEL_UP]) and self._want_level(env, p):
                return A_LEVEL_UP

            # --- Best buy (consider all shop slots, even if board full).
            slot, best_minion, best_score = self._find_best_in_shop(p, r)
            buy_legal = (
                slot is not None
                and bool(mask[A_BUY_BASE + slot])
            )

            # Direct buy when there's room and the minion is good.
            if (
                slot is not None
                and best_minion is not None
                and not board_full(p)
                and buy_legal
                and best_score > 0.0
            ):
                return A_BUY_BASE + slot

            # Sell-then-buy when board full and incoming is clearly better.
            if (
                slot is not None
                and best_minion is not None
                and board_full(p)
                and best_score > 0.0
            ):
                # Use larger margin for non-mentor (mentor was handled above).
                sell_idx = self._best_sell_for_incoming(p, best_minion, mask, margin=3.0)
                if sell_idx is not None:
                    return A_SELL_BASE + sell_idx

            # --- Roll: when nothing strong in shop, gold available, under cap.
            if (
                bool(mask[A_ROLL])
                and p.gold >= 1
                and self._rolls[0] < self._max_rolls(p, r)
            ):
                # T3: roll for mentor/big_guy if shop lacks strong T3.
                if p.tavern_tier >= 3 and not self._shop_has_strong_t3(p):
                    self._rolls[0] += 1
                    return A_ROLL
                # Roll if shop has no current-tier minion (e.g., T2 shop showing
                # only T1 cards is wasted tempo).
                shop_tiers = [m.tier for _, m in shop_minions_with_slots(p.shop)]
                if shop_tiers and max(shop_tiers) < p.tavern_tier:
                    self._rolls[0] += 1
                    return A_ROLL
                # Generic: roll if shop best is mediocre and we can't level.
                # When already behind on HP in early T2, avoid gambling rolls —
                # buy whatever stabilizes the board.
                gap = self._hp_gap_favors_opponent(env, p)
                bleed_skip = (
                    self.early_hp_tempo
                    and r <= 5
                    and p.tavern_tier <= 2
                    and gap >= 5
                )
                if best_score < 6.0 and not bool(mask[A_LEVEL_UP]) and not bleed_skip:
                    self._rolls[0] += 1
                    return A_ROLL

            # Empty-board fallback: buy anything legal if we have nothing.
            if not p.board and slot is not None and buy_legal and not board_full(p):
                return A_BUY_BASE + slot

            return self._finish(env)
        finally:
            self._shop_context_env = None


# ----------------------------------------------------------------------------
# LookaheadBot: uses simulate_battle to score candidate buys against a
# synthesized "round-typical opponent" board.  More expensive, but should
# be stronger than purely card-priority based heuristics.
# ----------------------------------------------------------------------------


def _make_typical_opponent(round_number: int, rng: np.random.Generator) -> List[Minion]:
    """Synthesize a representative opponent board for the given round.

    Calibrated very loosely against fast_t3 / buffer_t2 trajectories.
    """
    from ..cards import make_minion as _mk

    if round_number <= 1:
        return [_mk("recruit")]
    if round_number == 2:
        return [_mk("recruit")]  # opponent that levelled
    if round_number == 3:
        return [_mk("recruit"), _mk("shield_bot")]
    if round_number == 4:
        return [_mk("recruit"), _mk("shield_bot"), _mk("pack_rat")]
    if round_number == 5:
        b = [_mk("shield_bot"), _mk("pack_rat"), _mk("bruiser")]
        # +1/+1 on shield_bot (typical buffer hit)
        b[0].bonus_attack += 1
        b[0].bonus_health += 1
        return b
    if round_number == 6:
        b = [_mk("shield_bot"), _mk("pack_rat"), _mk("bruiser"), _mk("buffer")]
        b[0].bonus_attack += 1
        b[0].bonus_health += 1
        b[1].bonus_attack += 1
        b[1].bonus_health += 1
        return b
    if round_number == 7:
        b = [_mk("shield_bot"), _mk("pack_rat"), _mk("big_guy"), _mk("summoner")]
        b[0].bonus_attack += 1
        b[0].bonus_health += 1
        return b
    # round_number >= 8 -> mid/late T3 board
    b = [_mk("big_guy"), _mk("commander"), _mk("summoner"), _mk("mentor")]
    for m in b[:2]:
        m.bonus_attack += 2
        m.bonus_health += 1
    return b


def _expected_dmg_advantage(
    my_board: List[Minion],
    opp_board: List[Minion],
    rng: np.random.Generator,
    n_battles: int = 16,
) -> float:
    """Return signed expected damage diff (mine - opp) over n_battles."""
    from copy import copy as _copy

    if not my_board and not opp_board:
        return 0.0
    diff = 0.0
    for _ in range(n_battles):
        mb = [_copy(m) for m in my_board]
        ob = [_copy(m) for m in opp_board]
        # Reset transient shield state (battle module handles this on entry,
        # but make sure shields are armed; make_minion sets has_shield).
        dmg_p0, dmg_p1 = simulate_battle(mb, ob, True, rng)
        # In simulate_battle: if I'm p0, dmg_p0 is damage done TO me, dmg_p1 is damage done TO opp.
        diff += float(dmg_p1) - float(dmg_p0)
    return diff / max(1, n_battles)


class LookaheadBot(HeuristicBot):
    """Lookahead bot: evaluates candidate buys/sells/rolls/levelups by
    simulating battles against a round-typical opponent board.

    For tractability, only one-step lookahead on the *current* shop phase.
    """

    name = "lookahead"
    order_style = "default"

    def __init__(self, n_battles: int = 12, seed: Optional[int] = None) -> None:
        super().__init__()
        self._n_battles = int(n_battles)
        self._rng = np.random.default_rng(seed if seed is not None else 0xA9EE)

    def _round(self, env: MiniBGEnv) -> int:
        return env.state.round_number

    def _eval_board(self, board: List[Minion], r: int) -> float:
        # Approximate "round in which this board will fight" = current round.
        # Use round+0 (we are still in the shopping phase, battle is right
        # after both players finish).
        opp = _make_typical_opponent(r, self._rng)
        return _expected_dmg_advantage(board, opp, self._rng, self._n_battles)

    def _candidate_buy_boards(
        self, p: PlayerState, mask: np.ndarray
    ) -> List[tuple[int, List[Minion]]]:
        """Return [(action, resulting_board), ...] for legal buys.

        Note: ON_BUY buffer effect picks a random target; we approximate
        by giving +1/+1 to the score-best non-buffer minion deterministically.
        """
        from copy import copy as _copy

        out: List[tuple[int, List[Minion]]] = []
        for slot, m in shop_minions_with_slots(p.shop):
            a = A_BUY_BASE + slot
            if not bool(mask[a]):
                continue
            new_board = [_copy(x) for x in p.board]
            new_minion = _copy(m)
            new_board.append(new_minion)
            # Fire buffer on_buy approximation
            if new_minion.card_id == "buffer":
                others = [
                    (i, score_minion_on_board(x))
                    for i, x in enumerate(new_board)
                    if x is not new_minion
                ]
                if others:
                    others.sort(key=lambda t: -t[1])
                    bidx = others[0][0]
                    new_board[bidx].bonus_attack += 1
                    new_board[bidx].bonus_health += 1
            out.append((a, new_board))
        return out

    def _candidate_sell_boards(
        self, p: PlayerState, mask: np.ndarray
    ) -> List[tuple[int, List[Minion]]]:
        from copy import copy as _copy

        out: List[tuple[int, List[Minion]]] = []
        for pos in range(len(p.board)):
            a = A_SELL_BASE + pos
            if not bool(mask[a]):
                continue
            nb = [_copy(x) for j, x in enumerate(p.board) if j != pos]
            out.append((a, nb))
        return out

    def choose_action(self, env: MiniBGEnv) -> int:
        mask = _mask(env)
        p = _me(env)
        self._sync(p)
        r = env.state.round_number

        # 1) Level-up: gated by simple round/health rules
        if (
            bool(mask[A_LEVEL_UP])
            and can_afford_level(p)
            and (
                (p.tavern_tier == 1 and r >= 2)
                or (
                    p.tavern_tier == 2
                    and r >= 4
                    and len(p.board) >= 3
                    and p.health >= 9
                )
                or (p.tavern_tier == 2 and r >= 6 and p.health >= 5)
            )
        ):
            return A_LEVEL_UP

        # 2) Score current board (no-op baseline = FINISH)
        baseline = self._eval_board(list(p.board), r)
        best_action: int = -1
        best_score: float = baseline - 0.05  # small bias to act if equal

        # 3) Score all candidate buys
        for a, nb in self._candidate_buy_boards(p, mask):
            if board_full(p):
                continue
            sc = self._eval_board(nb, r)
            if sc > best_score:
                best_score = sc
                best_action = a

        # 4) Score sell-then-buy combos when board full
        if board_full(p):
            from copy import copy as _copy

            best_buy_minion: Optional[Minion] = None
            best_buy_action: int = -1
            best_buy_score = -1e18
            for slot, m in shop_minions_with_slots(p.shop):
                if not bool(mask[A_BUY_BASE + slot]):
                    continue
                # Just look at the strongest incoming
                sc_buy = (
                    score_minion_on_board(m)
                    + (4.0 if Keyword.SHIELD in m.keywords else 0.0)
                    + (3.0 if any(ab.trigger == Trigger.ON_DEATH for ab in m.abilities) else 0.0)
                    + (4.0 if any(ab.trigger == Trigger.ON_TURN_END for ab in m.abilities) else 0.0)
                )
                if sc_buy > best_buy_score:
                    best_buy_score = sc_buy
                    best_buy_minion = m
                    best_buy_action = A_BUY_BASE + slot

            if best_buy_minion is not None:
                # Try each sell, evaluate resulting board (without the buy,
                # since we can only do one action this turn).
                for a_sell, nb_after_sell in self._candidate_sell_boards(p, mask):
                    sc = self._eval_board(nb_after_sell, r)
                    # Bias: prefer sell only if it sets up a meaningful upgrade
                    if sc > best_score - 0.1:
                        # Approximate "after buy next turn" by appending
                        nb_full = list(nb_after_sell) + [best_buy_minion]
                        sc_full = self._eval_board(nb_full, r)
                        if sc_full > best_score:
                            best_score = sc_full
                            best_action = a_sell

        # 5) Roll heuristic: only consider when we have spare gold and
        # the best buy seems weak.
        if (
            bool(mask[A_ROLL])
            and p.gold >= 1
            and self._rolls[0] < (4 if p.tavern_tier == 3 else 2)
        ):
            current_best_minion_score = -1e18
            for slot, m in shop_minions_with_slots(p.shop):
                if bool(mask[A_BUY_BASE + slot]):
                    current_best_minion_score = max(
                        current_best_minion_score, score_minion_on_board(m)
                    )
            # Only roll if shop is empty/poor and we are not better off
            # by just acting now.
            if current_best_minion_score < 6.0 and best_action == -1:
                self._rolls[0] += 1
                return A_ROLL

        if best_action == -1:
            return self._finish(env)
        return best_action


class ApexImprovedBot(ApexBot):
    """ApexBot with optional incremental improvements (toggleable via flags).

    Default: only ``improve_mentor`` and ``improve_shield`` are enabled, since
    A/B tests showed the other three are within noise (kept as flags for
    documentation / experimentation; they only affect ``_buy_score`` and so
    rarely trigger sell-then-buy paths in late game).

      improve_mentor:   value mentor's ON_TURN_END snowball by adding a
                        rounds-left bonus to its on-board AND buy scores so
                        we stop accidentally selling mentor and start
                        swapping into it.  +5 pp vs apex.
      improve_shield:   shield re-arms every battle, so a SHIELD minion
                        prevents ~3-4 dmg per remaining battle.  Currently
                        ``score_minion_on_board(shield_bot) = 10`` (worst on
                        a T3 board), so shield_bot gets sold at R7+
                        (frequency drops 55% -> 17% between R6 and R7).
                        Add rounds-left bonus so a buffed shield_bot is kept.

    These three are kept disabled (no measurable effect, see analysis):
      improve_gold_leak / improve_buffer / improve_commander -- these only
      tweak ``_buy_score``; in late game the board is full and decisions go
      through ``_best_sell_for_incoming`` which uses the on-board score, so
      the buy-side tweaks rarely fire.

    Optional tweaks (A/B / ladder experiments):
      tweak_cmdr_stack:    penalize stacked commanders in ``_buy_score`` and
                           ``_board_score`` (sell-then-buy path).
      tweak_mentor_guard:  skip forced mentor buy/sell when HP is low, strong
                           targets < 2, or board already has 2+ mentors.
    """

    name = "apex_improved"
    order_style = "default"

    def __init__(
        self,
        *,
        improve_mentor: bool = True,
        improve_shield: bool = True,
        improve_gold_leak: bool = False,
        improve_buffer: bool = False,
        improve_commander: bool = False,
        tweak_cmdr_stack: bool = False,
        tweak_mentor_guard: bool = True,
    ) -> None:
        super().__init__()
        self.improve_mentor = improve_mentor
        self.improve_shield = improve_shield
        self.improve_gold_leak = improve_gold_leak
        self.improve_buffer = improve_buffer
        self.improve_commander = improve_commander
        self.tweak_cmdr_stack = tweak_cmdr_stack
        self.tweak_mentor_guard = tweak_mentor_guard

    def _uses_round_aware_scoring(self) -> bool:
        return (
            self.improve_mentor
            or self.improve_shield
            or self.improve_gold_leak
            or self.tweak_cmdr_stack
        )

    def _mentor_buy_priority_ok(self, p: PlayerState, r: int) -> bool:
        if not self.tweak_mentor_guard:
            return super()._mentor_buy_priority_ok(p, r)
        nm = sum(1 for x in p.board if x.card_id == "mentor")
        if nm >= 2:
            return False
        if nm >= 1 and self._strong_targets(p) < 2:
            return False
        if nm >= 1 and p.health <= 10:
            return False
        return True

    # --- gold-leak: more rolls and less strict sell margin late-game ---
    def _max_rolls(self, p: PlayerState, r: int) -> int:
        base = super()._max_rolls(p, r)
        if self.improve_gold_leak and p.tavern_tier >= 3 and r >= 7:
            return 6
        return base

    # Round-aware board score for swap decisions.
    def _board_score(self, m: Minion, r: int, p: PlayerState) -> float:
        s = score_minion_on_board(m)
        rounds_left = max(0, 13 - r)
        if self.tweak_cmdr_stack and m.card_id == "commander":
            n_cmd = sum(1 for x in p.board if x.card_id == "commander")
            on_board = any(m is mm for mm in p.board)
            if not on_board:
                n_cmd += 1
            if n_cmd >= 2:
                s -= 5.0 * (n_cmd - 1)
        if self.improve_mentor and m.card_id == "mentor":
            s += rounds_left * 1.6
        if self.improve_shield and Keyword.SHIELD in m.keywords:
            # Shield re-arms each battle: ~3-4 dmg absorbed per future battle.
            s += rounds_left * 0.8
            # Buffed shield_bot is far more valuable: it deals more damage
            # AND its survivor stats matter for next-round attack.
            buff_total = m.bonus_attack + m.bonus_health
            if buff_total >= 2:
                s += min(buff_total, 6) * 0.8
        return s

    def _best_sell_for_incoming(
        self,
        p: PlayerState,
        incoming: Minion,
        mask: np.ndarray,
        margin: float = 3.0,
    ) -> Optional[int]:
        if not p.board:
            return None
        if p.gold < BUY_COST - 1:
            return None

        r = getattr(self, "_round_for_sell", None)
        if r is None or not self._uses_round_aware_scoring():
            return super()._best_sell_for_incoming(p, incoming, mask, margin)

        my_scores = [(i, self._board_score(m, r, p)) for i, m in enumerate(p.board)]
        incoming_score = self._board_score(incoming, r, p)
        my_scores.sort(key=lambda t: t[1])
        worst_idx, worst_score = my_scores[0]

        eff_margin = margin
        if self.improve_gold_leak and r >= 7:
            eff_margin = max(0.5, margin - 1.5)

        if incoming_score - worst_score < eff_margin:
            return None
        if not bool(mask[A_SELL_BASE + worst_idx]):
            return None
        return worst_idx

    # --- buy_score adjustments ---
    def _buy_score(self, m: Minion, p: PlayerState, r: int) -> float:
        s = super()._buy_score(m, p, r)

        if self.improve_mentor and m.card_id == "mentor":
            tgt = self._strong_targets(p)
            if tgt >= 1:
                rounds_left = max(0, 13 - r)
                s += rounds_left * 1.0

        if self.improve_shield and Keyword.SHIELD in m.keywords:
            # Buying shield_bot late-game has incremental value from
            # remaining-battle absorbs.
            rounds_left = max(0, 13 - r)
            s += rounds_left * 0.5

        if self.improve_buffer and m.card_id == "buffer":
            non_buffer = sum(1 for x in p.board if x.card_id != "buffer")
            if non_buffer >= 1:
                tgt = self._strong_targets(p)
                rounds_left = max(0, 13 - r)
                if tgt >= 2 and r >= 5:
                    s += rounds_left * 0.8

        if (self.improve_commander or self.tweak_cmdr_stack) and m.card_id == "commander":
            n_cmd = sum(1 for x in p.board if x.card_id == "commander")
            if n_cmd >= 1:
                s -= 5.0 * n_cmd

        return s

    def choose_action(self, env: MiniBGEnv) -> int:
        # Stash round so _best_sell_for_incoming (called from parent) can
        # access it without a signature change.
        self._round_for_sell = env.state.round_number
        try:
            return super().choose_action(env)
        finally:
            self._round_for_sell = None


def default_bot_constructors() -> dict[str, type[HeuristicBot]]:
    return {
        "random": RandomBot,
        "tempo": TempoBot,
        "buffer_t1": BufferT1Bot,
        "wide_t1": WideT1Bot,
        "early_t2_pressure": EarlyT2PressureBot,
        "delayed_t2_pressure": DelayedT2PressureBot,
        "balanced": BalancedBot,
        "token": TokenBot,
        "fast_t3": FastT3Bot,
        "t3_scaler": T3ScalerBot,
        "buffer_t2": BufferT2Bot,
        "buffer_t2_t3": BufferT2T3Bot,
        "t2_into_t3_pressure": T2IntoT3PressureBot,
        "apex": ApexBot,
        "apex_improved": ApexImprovedBot,
        "lookahead": LookaheadBot,
    }
