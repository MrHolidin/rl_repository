"""MiniBG-style heuristic bots adapted to BGLike action indices."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np

from ..action_map import (
    A_BUY_BASE,
    A_DISCOVER_BASE,
    A_FINISH,
    A_LEVEL_UP,
    A_PLACE_BASE,
    A_ROLL,
    A_SELL_BASE,
    buy_slot,
    is_buy,
    is_discover_pick,
    is_magnet,
    is_sell,
    is_swap_board,
)
from ..actions import BOARD_SIZE, BUY_COST, HAND_SIZE, MAX_SHOP_SLOTS, SELL_REWARD
from src.bg_lobby.player import PlayerState

from .common import (
    legal_env_indices,
    masked_finish,
    pick_rl_apply_action,
)
from .env_view import BGLikeHeuristicEnvView

HeuristicEnv = Union[BGLikeHeuristicEnvView]


def _mask(env: HeuristicEnv) -> np.ndarray:
    return env.legal_actions_mask


def _me(env: HeuristicEnv) -> PlayerState:
    seat = env.current_player()
    return env.state.players[seat]


def _maybe_rl_apply(env: HeuristicEnv, mask: np.ndarray) -> Optional[int]:
    return pick_rl_apply_action(env, mask)


def _mandatory_place_actions(mask: np.ndarray, p: PlayerState) -> list[int]:
    out: list[int] = []
    for i in range(HAND_SIZE):
        if i < len(p.hand) and p.hand[i] is not None and bool(mask[A_PLACE_BASE + i]):
            out.append(A_PLACE_BASE + i)
    return out


def _replacement_sell_actions(
    mask: np.ndarray,
    p: PlayerState,
    *,
    tier_up_style: bool,
) -> list[int]:
    if len(p.board) < BOARD_SIZE:
        return []
    gold_after = p.gold + SELL_REWARD
    hand_has = any(
        p.hand[i] is not None for i in range(min(HAND_SIZE, len(p.hand)))
    )
    tier = p.tavern_tier
    out: list[int] = []

    def has_policy_buy() -> bool:
        for s in range(MAX_SHOP_SLOTS):
            a = A_BUY_BASE + s
            if not bool(mask[a]):
                continue
            m = p.shop[s]
            if m is None:
                continue
            if tier_up_style:
                if m.tier == tier:
                    return True
            elif m.tier == 1:
                return True
        return False

    shop_replace = gold_after >= BUY_COST and has_policy_buy()

    for i in range(len(p.board)):
        sa = A_SELL_BASE + i
        if not bool(mask[sa]):
            continue
        bm = p.board[i]
        if hand_has:
            out.append(sa)
            continue
        if not shop_replace:
            continue
        if tier_up_style and bm.tier >= tier:
            continue
        out.append(sa)
    return out


def _prefer_growth_then_finish(
    mask: np.ndarray,
    p: PlayerState,
    allowed: list[int],
    rng: np.random.Generator,
) -> int:
    allowed = [a for a in allowed if not is_swap_board(a)]
    buys = [a for a in allowed if is_buy(a)]
    if buys:
        return int(rng.choice(buys))
    if bool(mask[A_ROLL]) and A_ROLL in allowed and len(p.board) < BOARD_SIZE:
        return A_ROLL
    mags = [a for a in allowed if is_magnet(a)]
    if mags:
        return int(rng.choice(mags))
    non_fin = [a for a in allowed if a != A_FINISH]
    if non_fin:
        return int(rng.choice(non_fin))
    if A_FINISH in allowed:
        return A_FINISH
    return int(rng.choice(allowed)) if allowed else masked_finish(mask)


class HeuristicBot(ABC):
    name: str = "heuristic"
    order_style: str = "default"

    def _finish(self, env: HeuristicEnv) -> int:
        mask = _mask(env)
        p = _me(env)

        for i in range(3):
            if bool(mask[A_DISCOVER_BASE + i]):
                return A_DISCOVER_BASE + i

        for i in range(HAND_SIZE):
            if i < len(p.hand) and bool(mask[A_PLACE_BASE + i]):
                return A_PLACE_BASE + i

        return masked_finish(mask)

    def choose_action(self, env: HeuristicEnv) -> int:
        return self._choose_action(env)

    @abstractmethod
    def _choose_action(self, env: HeuristicEnv) -> int:
        raise NotImplementedError


class Tier1RandomBot(HeuristicBot):
    name = "t1_random"

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def _choose_action(self, env: HeuristicEnv) -> int:
        mask = _mask(env)
        p = _me(env)
        rl_apply = _maybe_rl_apply(env, mask)
        if rl_apply is not None:
            return rl_apply
        legal = legal_env_indices(mask)
        disc = [a for a in legal if is_discover_pick(a)]
        if disc:
            return int(self._rng.choice(disc))

        places = _mandatory_place_actions(mask, p)
        if places:
            return int(self._rng.choice(places))

        ok_sell = set(_replacement_sell_actions(mask, p, tier_up_style=False))
        allowed: list[int] = []
        for a in legal:
            if a == A_LEVEL_UP:
                continue
            if is_buy(a):
                m = p.shop[buy_slot(a)]
                if m is None or m.tier != 1:
                    continue
            if is_sell(a) and a not in ok_sell:
                continue
            allowed.append(a)

        if not allowed:
            return self._finish(env)
        return _prefer_growth_then_finish(mask, p, allowed, self._rng)


class TierUpRandomBot(HeuristicBot):
    name = "t_up_random"

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    def _choose_action(self, env: HeuristicEnv) -> int:
        mask = _mask(env)
        p = _me(env)
        rl_apply = _maybe_rl_apply(env, mask)
        if rl_apply is not None:
            return rl_apply
        legal = legal_env_indices(mask)
        disc = [a for a in legal if is_discover_pick(a)]
        if disc:
            return int(self._rng.choice(disc))

        places = _mandatory_place_actions(mask, p)
        if places:
            return int(self._rng.choice(places))

        if bool(mask[A_LEVEL_UP]) and self._rng.random() < 0.25:
            return A_LEVEL_UP

        tier = p.tavern_tier
        high_buys = [
            a
            for a in legal
            if is_buy(a)
            and p.shop[buy_slot(a)] is not None
            and p.shop[buy_slot(a)].tier == tier
        ]
        if high_buys:
            return int(self._rng.choice(high_buys))

        low_buys = [
            a
            for a in legal
            if is_buy(a)
            and (m := p.shop[buy_slot(a)]) is not None
            and m.tier < tier
        ]
        if low_buys:
            return int(self._rng.choice(low_buys))

        ok_sell = set(_replacement_sell_actions(mask, p, tier_up_style=True))
        pool = [a for a in legal if not is_sell(a) or a in ok_sell]
        if not pool:
            return self._finish(env)
        return _prefer_growth_then_finish(mask, p, pool, self._rng)


def default_bot_constructors() -> dict[str, type[HeuristicBot]]:
    from .structured_bot import StructuredHeuristicBot

    return {
        Tier1RandomBot.name: Tier1RandomBot,
        TierUpRandomBot.name: TierUpRandomBot,
        StructuredHeuristicBot.name: StructuredHeuristicBot,
    }


def make_bot(name: str, seed: Optional[int] = None) -> HeuristicBot:
    ctors = default_bot_constructors()
    if name not in ctors:
        raise ValueError(f"unknown bglike bot {name!r}; valid: {sorted(ctors)}")
    return ctors[name](seed=seed)


__all__ = [
    "HeuristicBot",
    "Tier1RandomBot",
    "TierUpRandomBot",
    "default_bot_constructors",
    "make_bot",
]
