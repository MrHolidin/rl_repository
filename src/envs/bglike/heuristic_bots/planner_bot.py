"""Planner heuristic bot for BGLike — every decision derived from a target board.

``structured``/``elemental`` score single actions off a priority list, so a buy
never knows who it displaces and a sell never knows who is waiting for the slot.
This bot instead answers one question per step:

    what is the best set of <=7 minions I can hold at the end of this turn,
    given board + hand + affordable shop offers?

That set (``_target_board``) is a small knapsack over the 7 board slots with the
gold budget as the constraint (a buy costs ``BUY_COST``, a displaced board minion
refunds its sell value). Buying, selling and playing are then just the cheapest
legal step toward the set, which makes replacement value explicit:

* a shop minion is bought only if it beats the board minion it would displace;
* a board minion is sold only when something better is waiting for its slot;
* a card in hand is played when its slot is worth more to it than to anyone else,
  and is otherwise left in hand (or dumped onto the board and later sold, which
  falls out of the same rule once the hand starts blocking buys).

Leveling and the roll bar are deliberately taken over from ``structured`` so the
comparison isolates the planning change.

Scoring the set as a set (auras priced against the minions that actually carry
them, spent battlecries not re-counted) was tried and measured: it changes the
chosen set in ~11% of decisions and is worth -0.10 +- 0.23 places over 200
games, i.e. nothing for the code it costs. The set choice is near-forced —
~10 candidates for 7 slots, 6-7 of which are already on the board — so the
objective's resolution is not what limits this bot.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..action_map import (
    A_BUY_BASE,
    A_DISCOVER_BASE,
    A_FINISH_FREEZE_SHOP,
    A_LEVEL_UP,
    A_PLACE_BASE,
    A_ROLL,
    A_SELL_BASE,
    is_discover_pick,
    is_magnet,
    magnet_hand_board,
)
from ..actions import (
    BOARD_SIZE,
    HAND_SIZE,
    MAX_SHOP_ACTIONS,
    MAX_SHOP_SLOTS,
    MAX_TIER,
)
from src.bg_catalog.cards import make_minion
from src.bg_core.minion import Minion
from src.bg_lobby.player import PendingChoiceKind, PlayerState
from src.bg_recruitment.economy import (
    effective_buy_cost,
    effective_roll_cost,
    effective_sell_reward,
)
from src.envs.minibg.heuristic_bots.value_model import (
    adapt_choice_score,
    board_power,
    dominant_race,
    minion_shop_value,
    roll_threshold_adjusted,
    rounds_left_estimate,
    triple_cluster_keep_bonus_board,
    triple_progress_buy_bonus,
    triple_progress_place_bonus,
)

from .bots import HeuristicBot, HeuristicEnv, _mask, _me
from .common import legal_env_indices, masked_finish, pick_rl_apply_action

# Flat stand-in for "value of the act of playing a minion" (board reactions to
# AFTER_FRIENDLY_MINION_PLACED, own battlecry beyond what the stat model sees).
# Triple progress is scored separately; this is the residual.
V_PLAY_FLAT = 0.6

# Keep the shop budget clear of the hard engine cap so a turn can always finish.
SHOP_ACTION_RESERVE = 4

SRC_BOARD = "b"
SRC_HAND = "h"
SRC_SHOP = "s"

# What one more refresh is worth, in this bot's own value units. Sampled from
# post-roll shops over 25 lobbies (an unconditioned draw of the offer pool):
# ``_ROLL_BENEFIT[tier][i]`` = E[max(X - cur, 0)] for X the best offer of a fresh
# shop and ``cur`` = ``_ROLL_CUR_GRID[i]`` the best offer already in front of us.
_ROLL_CUR_GRID = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
_ROLL_BENEFIT = {
    1: (9.69, 5.38, 3.05, 1.34, 0.27, 0.05),
    2: (9.15, 4.23, 1.40, 0.60, 0.19, 0.05),
    3: (11.76, 6.77, 2.74, 1.09, 0.41, 0.07),
    4: (15.23, 10.23, 5.45, 2.58, 1.05, 0.25),
    5: (16.60, 11.60, 6.75, 3.21, 1.19, 0.50),
    6: (18.32, 13.32, 8.36, 4.30, 1.76, 0.72),
}
# Mean best offer of a fresh shop — what next turn is worth without a freeze.
_FRESH_BEST_MEAN = {1: 14.69, 2: 14.15, 3: 16.76, 4: 20.23, 5: 21.60, 6: 23.32}

# A refresh has to be able to turn up something; below this it is just spending.
_ROLL_MIN_BENEFIT = 0.5


def roll_benefit(tier: int, cur: float) -> float:
    """Expected gain of one refresh when the best offer in hand is worth ``cur``."""
    row = _ROLL_BENEFIT[max(1, min(6, int(tier)))]
    if cur <= _ROLL_CUR_GRID[0]:
        return row[0] + (_ROLL_CUR_GRID[0] - cur)
    if cur >= _ROLL_CUR_GRID[-1]:
        return max(0.0, row[-1] - (cur - _ROLL_CUR_GRID[-1]) * 0.02)
    for i in range(len(_ROLL_CUR_GRID) - 1):
        lo, hi = _ROLL_CUR_GRID[i], _ROLL_CUR_GRID[i + 1]
        if lo <= cur <= hi:
            t = (cur - lo) / (hi - lo)
            return row[i] * (1 - t) + row[i + 1] * t
    return 0.0


@dataclass
class _Cand:
    """One minion that could occupy a board slot at end of turn."""

    src: str
    idx: int
    m: Minion
    v: float


def _strongest_alive_opponent_board_power(
    env: HeuristicEnv, seat: int, rl: int, rn: int
) -> float:
    best = 0.0
    for o in env.state.alive:
        if o == seat:
            continue
        bp = board_power(env.state.players[o].board, rounds_left=rl, round_number=rn)
        best = max(best, bp)
    return best


def _hand_free_slots(p: PlayerState) -> int:
    return sum(1 for s in p.hand if s is None)


class PlannerHeuristicBot(HeuristicBot):
    name = "planner"

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Valuation
    # ------------------------------------------------------------------

    def _v(self, m: Minion, *, board_len: int, dominant, rl: int, rn: int) -> float:
        """Value of a minion standing on the board.

        ``tavern_tier_cap`` is deliberately left out. That term (+3.4 + 0.22*tier
        for a minion of your own tier) is a *shop preference*, and using it as
        board value re-ranks the whole board every time the tavern tier changes,
        which sends the bot off to reshuffle a board that did not get worse.
        Dropping it is worth -0.209 +- 0.130 places over 600 8-player games.
        """
        return minion_shop_value(
            m,
            rounds_left=rl,
            dominant=dominant,
            board_len=max(1, board_len),
            round_number=rn,
        )

    def _candidates(self, p: PlayerState, rl: int, rn: int) -> List[_Cand]:
        """Everything that could stand on the board at end of turn, valued once."""
        dom = dominant_race(p.board)
        bl = len(p.board)
        out: List[_Cand] = []

        for i, m in enumerate(p.board):
            v = self._v(m, board_len=bl, dominant=dom, rl=rl, rn=rn)
            v += triple_cluster_keep_bonus_board(p, i)
            out.append(_Cand(SRC_BOARD, i, m, v))

        for i in range(min(HAND_SIZE, len(p.hand))):
            m = p.hand[i]
            if m is None or m.is_triple_reward_spell:
                continue
            v = self._v(m, board_len=bl + 1, dominant=dom, rl=rl, rn=rn)
            v += triple_progress_place_bonus(p, m.card_id, i) + V_PLAY_FLAT
            out.append(_Cand(SRC_HAND, i, m, v))

        for s in range(min(MAX_SHOP_SLOTS, len(p.shop))):
            m = p.shop[s]
            if m is None:
                continue
            v = self._v(m, board_len=bl + 1, dominant=dom, rl=rl, rn=rn)
            v += triple_progress_buy_bonus(p, m.card_id) + V_PLAY_FLAT
            out.append(_Cand(SRC_SHOP, s, m, v))

        out.sort(key=lambda c: -c.v)
        return out

    def _plan_cost(self, p: PlayerState, sel: List[_Cand]) -> int:
        """Gold needed for a selection: buys minus the refunds of displaced minions."""
        buy_cost = effective_buy_cost(p)
        n_buy = sum(1 for c in sel if c.src == SRC_SHOP)
        kept = {c.idx for c in sel if c.src == SRC_BOARD}
        refund = sum(
            effective_sell_reward(m)
            for i, m in enumerate(p.board)
            if i not in kept
        )
        return buy_cost * n_buy - refund

    def _select(self, p: PlayerState, cands: List[_Cand], max_buys: int) -> List[_Cand]:
        sel: List[_Cand] = []
        n_buy = 0
        for c in cands:
            if len(sel) >= BOARD_SIZE:
                break
            if c.src == SRC_SHOP:
                if n_buy >= max_buys:
                    continue
                n_buy += 1
            sel.append(c)
        return sel

    def _target_board(
        self, p: PlayerState, cands: List[_Cand]
    ) -> Tuple[List[_Cand], List[_Cand]]:
        """Best affordable set of <=7 end-of-turn minions; returns (chosen, dropped)."""
        buy_cost = max(1, effective_buy_cost(p))
        ceiling = min(
            sum(1 for c in cands if c.src == SRC_SHOP),
            (p.gold + len(p.board)) // buy_cost + 1,
            BOARD_SIZE,
        )
        chosen = self._select(p, cands, 0)
        for max_buys in range(ceiling, 0, -1):
            sel = self._select(p, cands, max_buys)
            if self._plan_cost(p, sel) <= p.gold:
                chosen = sel
                break
        keys = {(c.src, c.idx) for c in chosen}
        dropped = [c for c in cands if (c.src, c.idx) not in keys]
        return chosen, dropped

    # ------------------------------------------------------------------
    # Sub-decisions carried over from ``structured`` (unchanged on purpose)
    # ------------------------------------------------------------------

    def _pick_discover(self, env: HeuristicEnv, mask: np.ndarray, p: PlayerState) -> int:
        pc = p.pending_choice
        assert pc is not None
        rl = rounds_left_estimate(env.state.round_number)
        rn = env.state.round_number
        best_a = -1
        best_s = -1e18
        for i in range(3):
            a = A_DISCOVER_BASE + i
            if not bool(mask[a]):
                continue
            tok = pc.options[i]
            if pc.kind in (
                PendingChoiceKind.DISCOVER_MURLOC,
                PendingChoiceKind.TRIPLE_REWARD_DISCOVER,
            ):
                patch = env._game._patch if hasattr(env, "_game") else env.patch
                m = make_minion(tok, patch=patch)
                sc = self._v(
                    m,
                    board_len=len(p.board) + 1,
                    dominant=dominant_race(p.board),
                    rl=rl,
                    rn=rn,
                )
            else:
                sc = adapt_choice_score(tok)
            if sc > best_s:
                best_s = sc
                best_a = a
        assert best_a >= 0
        return int(best_a)

    def _magnet_delta(self, p: PlayerState, rl: int, rn: int, env_action: int) -> float:
        hi, bi = magnet_hand_board(env_action)
        hm = p.hand[hi]
        bm = p.board[bi]
        if hm is None or bm is None:
            return -1e18
        dom = dominant_race(p.board)
        bl = len(p.board)
        v_before = self._v(bm, board_len=bl, dominant=dom, rl=rl, rn=rn)
        v_hand = self._v(hm, board_len=bl, dominant=dom, rl=rl, rn=rn)
        t = copy(bm)
        mg = copy(hm)
        from src.bg_recruitment.place import merge_magnetic_inplace

        merge_magnetic_inplace(t, mg)
        v_after = self._v(t, board_len=bl, dominant=dom, rl=rl, rn=rn)
        return v_after - v_before - v_hand

    def _should_level_up(self, env: HeuristicEnv, p: PlayerState, rl: int) -> bool:
        if p.tavern_tier >= MAX_TIER:
            return False
        if p.gold < p.next_tier_up_cost:
            return False
        rn = env.state.round_number
        seat = env.current_player()
        mine_bp = board_power(p.board, rounds_left=rl, round_number=rn)
        opp_bp = _strongest_alive_opponent_board_power(env, seat, rl, rn)
        gold_left = p.gold - p.next_tier_up_cost
        cost = p.next_tier_up_cost
        ratio = mine_bp / max(opp_bp, 1e-6)
        if rn <= 10 and ratio < 0.62:
            return False
        if ratio >= 1.10 and gold_left >= 1:
            return True
        if cost <= 4 and rn >= 5 and ratio >= 0.70 and gold_left >= 2:
            return True
        if rn >= 7 and ratio >= 0.85 and gold_left >= 3:
            return True
        if rn >= 11 and ratio >= 0.76 and gold_left >= 2:
            return True
        if gold_left >= 4 and ratio >= 0.72:
            return True
        return False

    def _finish(self, env: HeuristicEnv) -> int:
        mask = _mask(env)
        p = _me(env)
        if (
            p.pending_choice is not None
            and p.pending_choice.kind == PendingChoiceKind.TRANSFORM_SHOP_MINION
        ):
            buys = [
                A_BUY_BASE + s
                for s in range(MAX_SHOP_SLOTS)
                if bool(mask[A_BUY_BASE + s])
            ]
            if buys:
                return int(buys[0])
        for i in range(3):
            if bool(mask[A_DISCOVER_BASE + i]):
                return A_DISCOVER_BASE + i
        for i in range(min(HAND_SIZE, len(p.hand))):
            if bool(mask[A_PLACE_BASE + i]):
                return A_PLACE_BASE + i
        if self._should_freeze(env, p, mask):
            return A_FINISH_FREEZE_SHOP
        return masked_finish(mask)

    # ------------------------------------------------------------------
    # Main decision
    # ------------------------------------------------------------------

    def _choose_action(self, env: HeuristicEnv) -> int:
        mask = _mask(env)
        p = _me(env)

        rl_apply = pick_rl_apply_action(env, mask)
        if rl_apply is not None:
            return rl_apply

        legal = legal_env_indices(mask)
        if any(is_discover_pick(a) for a in legal):
            return self._pick_discover(env, mask, p)

        rl = rounds_left_estimate(env.state.round_number)
        rn = env.state.round_number

        # A triple reward spell is free value and does not cost a board slot.
        for i in range(min(HAND_SIZE, len(p.hand))):
            hm = p.hand[i]
            if hm is not None and hm.is_triple_reward_spell and bool(mask[A_PLACE_BASE + i]):
                return A_PLACE_BASE + i

        magnets = [a for a in legal if is_magnet(a)]
        if magnets:
            ranked = sorted(
                magnets, key=lambda a: self._magnet_delta(p, rl, rn, a), reverse=True
            )
            if self._magnet_delta(p, rl, rn, ranked[0]) >= -1.15:
                return int(ranked[0])

        if bool(mask[A_LEVEL_UP]) and self._should_level_up(env, p, rl):
            return A_LEVEL_UP

        if p.shop_actions_used >= MAX_SHOP_ACTIONS - SHOP_ACTION_RESERVE:
            return self._finish(env)

        cands = self._candidates(p, rl, rn)
        chosen, dropped = self._target_board(p, cands)

        want_hand = [c for c in chosen if c.src == SRC_HAND]
        want_shop = [c for c in chosen if c.src == SRC_SHOP]
        drop_board = [c for c in dropped if c.src == SRC_BOARD]
        free_slots = BOARD_SIZE - len(p.board)

        # 1. Put a wanted card from hand onto a free slot.
        if free_slots > 0 and want_hand:
            for c in want_hand:
                if bool(mask[A_PLACE_BASE + c.idx]):
                    return A_PLACE_BASE + c.idx

        # 2. Free a slot for something the plan wants — and only then.
        if free_slots == 0 and (want_hand or want_shop) and drop_board:
            worst = min(drop_board, key=lambda c: c.v)
            if bool(mask[A_SELL_BASE + worst.idx]):
                return A_SELL_BASE + worst.idx

        # 3. Buy — unless a refresh is the better use of the same gold.
        if want_shop:
            best = want_shop[0]
            displaced = min((c.v for c in drop_board), default=0.0)
            marginal = best.v - (displaced if free_slots == 0 else 0.0)
            if self._should_roll(env, p, mask, legal, rl, rn, marginal):
                return A_ROLL
            if _hand_free_slots(p) == 0:
                relief = self._unclog_hand(mask, p, chosen, drop_board, free_slots)
                if relief is not None:
                    return relief
            if bool(mask[A_BUY_BASE + best.idx]):
                return A_BUY_BASE + best.idx

        # 4. Nothing worth buying: refresh while the gold can still be spent.
        if self._should_roll(env, p, mask, legal, rl, rn, None):
            return A_ROLL

        # 5. Park leftover hand cards on free slots rather than end with them idle.
        if free_slots > 0:
            for c in cands:
                if c.src == SRC_HAND and bool(mask[A_PLACE_BASE + c.idx]):
                    return A_PLACE_BASE + c.idx

        return self._finish(env)

    def _best_offer_value(self, p: PlayerState, rl: int, rn: int) -> float:
        dom = dominant_race(p.board)
        bl = len(p.board) + 1
        return max(
            (
                self._v(m, board_len=bl, dominant=dom, rl=rl, rn=rn)
                for m in p.shop
                if m is not None
            ),
            default=0.0,
        )

    def _should_freeze(self, env: HeuristicEnv, p: PlayerState, mask: np.ndarray) -> bool:
        """Keep this shop only if it beats spending the leftover gold on refreshes.

        Freezing and refreshing are one decision, not two: on their own each is
        worth nothing (+0.10 and +0.03 places over 200 games), because refreshing
        to zero gold with no way to bank the find is a waste, and freezing without
        having refreshed first just keeps the first shop that came along. Priced
        against each other they are worth -0.87 +- 0.22 places.
        """
        if not bool(mask[A_FINISH_FREEZE_SHOP]) or p.pending_choice is not None:
            return False
        rn = env.state.round_number
        best_offer = self._best_offer_value(p, rounds_left_estimate(rn), rn)
        tier = max(1, min(6, p.tavern_tier))
        delta = best_offer - _FRESH_BEST_MEAN[tier]
        if delta <= 0:
            return False
        return delta > roll_benefit(tier, best_offer) * (p.gold + 1)

    def _unclog_hand(
        self,
        mask: np.ndarray,
        p: PlayerState,
        chosen: List[_Cand],
        drop_board: List[_Cand],
        free_slots: int,
    ) -> Optional[int]:
        """Hand is full and blocks the buy: play the best card, or free a slot first."""
        hand_cands = [c for c in chosen if c.src == SRC_HAND]
        if free_slots > 0:
            for c in hand_cands:
                if bool(mask[A_PLACE_BASE + c.idx]):
                    return A_PLACE_BASE + c.idx
            for i in range(min(HAND_SIZE, len(p.hand))):
                if p.hand[i] is not None and bool(mask[A_PLACE_BASE + i]):
                    return A_PLACE_BASE + i
            return None
        if drop_board:
            worst = min(drop_board, key=lambda c: c.v)
            if bool(mask[A_SELL_BASE + worst.idx]):
                return A_SELL_BASE + worst.idx
        return None

    def _should_roll(
        self,
        env: HeuristicEnv,
        p: PlayerState,
        mask: np.ndarray,
        legal: List[int],
        rl: int,
        rn: int,
        marginal: Optional[float],
    ) -> bool:
        """Refresh when the shop's best marginal upgrade is below the tier bar."""
        if not bool(mask[A_ROLL]) or A_ROLL not in legal:
            return False
        roll_cost = effective_roll_cost(p)
        if p.free_roll_charges > 0 or roll_cost == 0:
            return True
        if marginal is None:
            # Nothing in the shop made the plan at all: only worth a refresh if
            # the gold left after it could still pay for a replacement.
            return p.gold - roll_cost >= self._replacement_cost(p)
        seat = env.current_player()
        mine_bp = board_power(p.board, rounds_left=rl, round_number=rn)
        opp_bp = _strongest_alive_opponent_board_power(env, seat, rl, rn)
        thr = roll_threshold_adjusted(
            tavern_tier=p.tavern_tier,
            round_number=rn,
            mine_board_power=mine_bp,
            opp_board_power=opp_bp,
        )
        if marginal >= thr:
            return False
        if p.gold - roll_cost >= self._replacement_cost(p):
            return True
        # Not enough gold left to buy afterwards — but gold does not survive the
        # turn, and a refresh can still turn up a shop worth freezing.
        if marginal is not None and marginal > 0:
            return False
        return (
            roll_benefit(p.tavern_tier, self._best_offer_value(p, rl, rn))
            > _ROLL_MIN_BENEFIT
        )

    def _replacement_cost(self, p: PlayerState) -> int:
        """Gold a buy still needs after this turn's refunds are counted in."""
        buy = effective_buy_cost(p)
        if len(p.board) >= BOARD_SIZE and p.board:
            return max(0, buy - min(effective_sell_reward(m) for m in p.board))
        return buy


__all__ = ["PlannerHeuristicBot"]
