from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from ..action_map import (
    A_BUY_BASE,
    A_LEVEL_UP,
    A_ROLL,
    A_SELECT_ORDER_BASE,
    A_SELL_BASE,
    NUM_ENV_ACTIONS,
)
from ..actions import MAX_TIER, SHOP_SIZE
from ..env import MiniBGEnv
from ..state import Minion, PlayerState

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
    legal_env_indices,
    order_key_default,
    order_key_token,
    reset_shop_counters_if_needed,
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
        mask = _mask(env)
        p = _me(env)
        key = order_key_token if self.order_style == "token" else order_key_default
        return choose_final_order(p.board, mask, key)

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
    }
