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
    is_swap_board,
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
from src.bg_core.effects import (
    AdaptAllMurlocsEffect,
    BuffAdjacentBattlecry,
    BuffAllOtherOfTribe,
    BuffTargetFriendlyBattlecry,
    DiscoverMurlocEffect,
    SummonEffect,
    SummonRandomMinionEffect,
    Trigger,
)
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PendingChoiceKind, PlayerState
from src.bg_recruitment.economy import (
    effective_buy_cost,
    effective_level_up_cost,
    effective_roll_cost,
    effective_sell_reward,
)
from src.envs.minibg.heuristic_bots.value_model import (
    adapt_choice_score,
    dominant_race,
    minion_shop_value,
    rounds_left_estimate,
    triple_cluster_keep_bonus_board,
    triple_progress_buy_bonus,
    triple_progress_place_bonus,
)

from .bots import HeuristicBot, HeuristicEnv, _mask, _me
from .common import (
    choose_one_swap_toward_target,
    legal_env_indices,
    masked_finish,
    pick_rl_apply_action,
)

# Flat stand-in for "value of the act of playing a minion" (board reactions to
# AFTER_FRIENDLY_MINION_PLACED, own battlecry beyond what the stat model sees).
# Triple progress is scored separately; this is the residual.
V_PLAY_FLAT = 0.6

# Keep the shop budget clear of the hard engine cap so a turn can always finish.
SHOP_ACTION_RESERVE = 4

SRC_BOARD = "b"
SRC_HAND = "h"
SRC_SHOP = "s"

# What one more refresh is worth, in this bot's own value units:
# ``_ROLL_BENEFIT[tier][i]`` = E[max(X - cur, 0)] for X the best offer of a fresh
# shop and ``cur`` = ``_ROLL_CUR_GRID[i]`` the best offer already on the counter.
#
# Drawn straight from the card pool with the round held fixed at 10 (4000 shops
# per cell, scripts/bglike_offer_calibration.py). Sampling this from live games
# instead — as the first version did — confounds tier with round, because a T5
# shop only ever appears at round 12+, and it came out overvaluing a refresh by
# 2-3x at the bars that matter late (T4 at bar 25: 1.05 sampled live vs 0.46
# actual). The bot then spent whole wallets refreshing a shop whose ceiling was
# below its own board: worth -0.955 +- 0.220 places over 200 games to fix.
_ROLL_CUR_GRID = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0)
_ROLL_BENEFIT = {
    1: (7.50, 3.50, 1.93, 0.70, 0.00, 0.00, 0.00, 0.00),
    2: (9.10, 4.25, 1.88, 0.66, 0.00, 0.00, 0.00, 0.00),
    3: (11.52, 6.53, 2.78, 1.12, 0.34, 0.05, 0.00, 0.00),
    4: (13.62, 8.62, 4.18, 1.70, 0.46, 0.04, 0.00, 0.00),
    5: (16.33, 11.33, 6.63, 3.42, 1.40, 0.49, 0.26, 0.05),
    6: (18.07, 13.07, 8.18, 4.23, 1.60, 0.40, 0.19, 0.04),
}
# Mean best offer of a fresh shop — what next turn is worth without a freeze.
_FRESH_BEST_MEAN = {1: 12.50, 2: 14.10, 3: 16.52, 4: 18.62, 5: 21.33, 6: 23.07}
# What one tier-up adds to the mean best offer, at a fixed round.
_TIER_DELTA = {1: 0.50, 2: 2.50, 3: 2.65, 4: 2.28, 5: 2.15}
# Buys per shop turn, measured over 633 planner turns in 10 lobbies.
_BUYS_PER_TURN = 1.2
# Mean stat line an adapt hands out, over the ten options.
_ADAPT_STATS = (2.0, 2.0)

# A refresh has to be able to turn up something; below this it is just spending.
_ROLL_MIN_BENEFIT = 0.5
# ...and it has to clear the board, not just the counter: once a fresh shop is
# not expected to beat the minion a purchase would displace by this much, the
# gold is better spent on the tier-up. Tried at 0.15 (-0.152) and 1.5 (+0.102)
# over 200 games; at 0.5 it is worth -0.155 +- 0.130 over 600.
_ROLL_FLOOR = 0.5


def roll_benefit(tier: int, cur: float) -> float:
    """Expected gain of one refresh when the best offer in hand is worth ``cur``."""
    row = _ROLL_BENEFIT[max(1, min(6, int(tier)))]
    if cur <= _ROLL_CUR_GRID[0]:
        return row[0] + (_ROLL_CUR_GRID[0] - cur)
    if cur >= _ROLL_CUR_GRID[-1]:
        return row[-1]
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


def _order_key(m: Minion) -> Tuple[float, str, int, int]:
    """Left-to-right combat order: attack first, deathrattles held back.

    The leftmost minion attacks first, so the board wants its trades made by the
    minions that win them; a deathrattle is worth more once there are bodies for
    what it summons. Compared against a taunt-first key (-0.185) and a
    chaff-in-front key (+0.043) over 200 games each.
    """
    w = -float(m.raw_attack) * 2.0 - float(m.max_health) * 0.3
    if any(ab.trigger == Trigger.ON_DEATH for ab in m.abilities):
        w += 15.0
    return (w, m.card_id, int(m.raw_attack), int(m.max_health))


def _buffed(m: Minion, atk: float, hp: float) -> Minion:
    out = copy(m)
    out.bonus_attack = m.bonus_attack + int(round(atk))
    out.bonus_health = m.bonus_health + int(round(hp))
    return out


def _tribe_match(m: Minion, tribe) -> bool:
    if tribe is None or tribe == Race.ALL:
        return True
    return m.race == tribe or m.race == Race.ALL


def _context_free_on_place(m: Minion) -> float:
    """What ``ability_shop_estimate`` charges for a battlecry, board or no board."""
    add = 0.0
    for ab in m.abilities:
        if ab.trigger != Trigger.ON_PLACE:
            continue
        eff = ab.effect
        if isinstance(eff, DiscoverMurlocEffect):
            add += 6.5 * float(eff.repeats)
        elif isinstance(eff, AdaptAllMurlocsEffect):
            add += 9.0 * float(eff.repeats)
        elif isinstance(eff, BuffAdjacentBattlecry):
            add += (eff.attack + eff.health) * 2.5 + (1.5 if eff.grant_taunt else 0.0)
        elif isinstance(eff, BuffTargetFriendlyBattlecry):
            add += (eff.attack + eff.health) * 1.6 * (
                0.75 if eff.filter_race is not None else 1.0
            )
        elif isinstance(eff, BuffAllOtherOfTribe):
            add += (eff.attack + eff.health) * 2.0
        else:
            add += 2.2
    return add


def _hand_free_slots(p: PlayerState) -> int:
    return sum(1 for s in p.hand if s is None)


class PlannerHeuristicBot(HeuristicBot):
    name = "planner"

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)
        # Only reachable through the env; needed to value a summon battlecry.
        self._patch = None

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
            v += self._play_value_correction(m, p, rl, rn, played=True)
            out.append(_Cand(SRC_BOARD, i, m, v))

        for i in range(min(HAND_SIZE, len(p.hand))):
            m = p.hand[i]
            if m is None or m.is_triple_reward_spell:
                continue
            v = self._v(m, board_len=bl + 1, dominant=dom, rl=rl, rn=rn)
            v += triple_progress_place_bonus(p, m.card_id, i) + V_PLAY_FLAT
            v += self._play_value_correction(m, p, rl, rn, played=False)
            out.append(_Cand(SRC_HAND, i, m, v))

        for s in range(min(MAX_SHOP_SLOTS, len(p.shop))):
            m = p.shop[s]
            if m is None:
                continue
            v = self._v(m, board_len=bl + 1, dominant=dom, rl=rl, rn=rn)
            v += triple_progress_buy_bonus(p, m.card_id) + V_PLAY_FLAT
            v += self._play_value_correction(m, p, rl, rn, played=False)
            out.append(_Cand(SRC_SHOP, s, m, v))

        out.sort(key=lambda c: -c.v)
        return out

    def _play_value_correction(
        self, m: Minion, p: PlayerState, rl: int, rn: int, *, played: bool
    ) -> float:
        """Replace the flat battlecry charge with the stats it actually moves.

        A battlecry hands out stats that then live in the targets' own bodies,
        so its value has to equal the increase it causes in the board total —
        price it higher and the same stats are paid for twice, once as the
        battlecry and once as the buffed target. As a difference it needs no
        constant, and the awkward cases price themselves: no targets of the
        tribe is zero, a full board makes a summon battlecry zero, and "buff a
        friendly" is worth its best actual target.

        A minion already on the board has spent its battlecry, so it keeps only
        the body. Worth -0.280 +- 0.129 places over 600 games.
        """
        cf = _context_free_on_place(m)
        if cf == 0.0:
            return 0.0
        if played:
            return -cf
        return self._delivered_play_value(m, p, rl, rn) - cf

    def _delivered_play_value(
        self, m: Minion, p: PlayerState, rl: int, rn: int
    ) -> float:
        total = 0.0
        free_slots = max(0, BOARD_SIZE - len(p.board) - 1)
        bl = len(p.board) + 1
        for ab in m.abilities:
            if ab.trigger != Trigger.ON_PLACE:
                continue
            eff = ab.effect
            if isinstance(eff, BuffAllOtherOfTribe):
                total += sum(
                    self._buff_delta(t, eff.attack, eff.health, bl, rl, rn)
                    for t in p.board
                    if _tribe_match(t, eff.tribe)
                )
            elif isinstance(eff, BuffTargetFriendlyBattlecry):
                targets = [
                    t
                    for t in p.board
                    if eff.filter_race is None or _tribe_match(t, eff.filter_race)
                ]
                if targets:
                    total += max(
                        self._buff_delta(t, eff.attack, eff.health, bl, rl, rn)
                        for t in targets
                    )
            elif isinstance(eff, BuffAdjacentBattlecry):
                deltas = sorted(
                    (
                        self._buff_delta(t, eff.attack, eff.health, bl, rl, rn)
                        for t in p.board
                    ),
                    reverse=True,
                )
                total += sum(deltas[:2])
            elif isinstance(eff, AdaptAllMurlocsEffect):
                total += float(eff.repeats) * sum(
                    self._buff_delta(t, _ADAPT_STATS[0], _ADAPT_STATS[1], bl, rl, rn)
                    for t in p.board
                    if _tribe_match(t, Race.MURLOC)
                )
            elif isinstance(eff, SummonEffect):
                if free_slots and self._patch is not None:
                    tok = make_minion(eff.token_id, patch=self._patch)
                    total += min(int(eff.count), free_slots) * self._v(
                        tok, board_len=bl, dominant=None, rl=rl, rn=rn
                    )
            elif isinstance(eff, SummonRandomMinionEffect):
                if free_slots:
                    tier = max(1, min(MAX_TIER, int(eff.exact_tier or 1)))
                    total += (
                        min(int(eff.count), free_slots) * _FRESH_BEST_MEAN[tier] * 0.6
                    )
            elif isinstance(eff, DiscoverMurlocEffect):
                tier = max(1, min(MAX_TIER, p.tavern_tier))
                total += float(eff.repeats) * _FRESH_BEST_MEAN[tier] * 0.5
            else:
                total += 2.2
        return total

    def _buff_delta(
        self, t: Minion, atk: float, hp: float, bl: int, rl: int, rn: int
    ) -> float:
        before = self._v(t, board_len=bl, dominant=None, rl=rl, rn=rn)
        after = self._v(_buffed(t, atk, hp), board_len=bl, dominant=None, rl=rl, rn=rn)
        return after - before

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

    def _level_roi(self, p: PlayerState, rn: int) -> float:
        """Board value a tier-up returns per gold, spread over the buys left."""
        if p.tavern_tier >= MAX_TIER:
            return 0.0
        cost = max(1, effective_level_up_cost(p))
        buys_left = max(0.0, rounds_left_estimate(rn) * _BUYS_PER_TURN)
        return _TIER_DELTA.get(p.tavern_tier, 0.0) * buys_left / cost

    def _displaced_value(self, p: PlayerState, rl: int, rn: int) -> float:
        """Value a purchase would have to push off the board (0 if a slot is free)."""
        if len(p.board) < BOARD_SIZE or not p.board:
            return 0.0
        return min(
            (c.v for c in self._candidates(p, rl, rn) if c.src == SRC_BOARD), default=0.0
        )

    def _board_roi(self, p: PlayerState, rl: int, rn: int) -> float:
        """Best board value per gold available to this turn — what a tier-up gives up."""
        cands = self._candidates(p, rl, rn)
        shop = [c for c in cands if c.src == SRC_SHOP]
        if not shop:
            return 0.0
        full = len(p.board) >= BOARD_SIZE
        worst = min((c.v for c in cands if c.src == SRC_BOARD), default=0.0) if full else 0.0
        need = max(1, self._replacement_cost(p))
        best = max(c.v for c in shop)
        gain = max(0.0, best - worst)
        roi = gain / need
        roll_cost = effective_roll_cost(p)
        if p.gold >= need + roll_cost:
            roi = max(roi, (gain + roll_benefit(p.tavern_tier, best)) / (need + roll_cost))
        return roi

    def _should_level_up(self, env: HeuristicEnv, p: PlayerState, rl: int) -> bool:
        """A tier-up is bought when it returns more per gold than this turn's board.

        It is the one purchase that pays in later turns — it lifts the whole offer
        distribution by ``_TIER_DELTA`` for every buy left in the game — so it
        belongs in the same currency as buying and refreshing. ``structured``'s
        board-power ratio never made that comparison, which is how the bot ended
        up at T4 with 9 gold refreshing a shop whose ceiling was below its own
        board: pricing it is worth -0.738 +- 0.222 places over 200 games.

        That ratio rule was kept on as a survival floor at first, on the reasoning
        that a tier-up buys nothing if the next combat ends the run. It does not
        hold up — removing it is worth a further -0.223 +- 0.130 over 600 games,
        and two other ways of pricing survival (discounting the tier-up by hit
        points, and weighting bodies and defensive keywords when close to death)
        measured +0.03 and +0.13. Being behind is a reason to reach for a better
        shop, not to stay in a worse one.
        """
        if p.tavern_tier >= MAX_TIER:
            return False
        if p.gold < effective_level_up_cost(p):
            return False
        rn = env.state.round_number
        return self._level_roi(p, rn) > self._board_roi(p, rl, rn)

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
        order = self._order_step(p, mask)
        if order is not None:
            return order
        if self._should_freeze(env, p, mask):
            return A_FINISH_FREEZE_SHOP
        return masked_finish(mask)

    def _order_step(self, p: PlayerState, mask: np.ndarray) -> Optional[int]:
        """Sort the board before the turn ends: big attackers first, deathrattles last.

        Board order decides combats and costs nothing — SWAP bypasses
        ``apply_action``, so it does not spend shop budget — and no bot in this
        package has ever issued one. Worth -0.432 +- 0.224 places over 200 games.
        """
        if p.pending_choice is not None or len(p.board) < 2:
            return None
        target = [
            p.board[i]
            for i in sorted(range(len(p.board)), key=lambda oi: _order_key(p.board[oi]))
        ]
        a = choose_one_swap_toward_target(p.board, mask, target)
        return int(a) if is_swap_board(a) else None

    # ------------------------------------------------------------------
    # Main decision
    # ------------------------------------------------------------------

    def _choose_action(self, env: HeuristicEnv) -> int:
        mask = _mask(env)
        p = _me(env)

        if self._patch is None:
            self._patch = getattr(env, "patch", None)

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
        """Refresh when the same gold returns more from a refresh than from a buy.

        A refresh does not add board value by itself — the find still has to be
        bought — so the comparison is *refresh then buy* against *buy now*, per
        gold spent, and not the refresh's own return (which always wins, since
        it costs 1 gold, and leaves the bot refreshing forever without buying).
        ``roll_threshold_adjusted``, an absolute bar borrowed from ``structured``
        and compared against a marginal gain, is what this replaces.
        """
        if not bool(mask[A_ROLL]) or A_ROLL not in legal:
            return False
        roll_cost = effective_roll_cost(p)
        if p.free_roll_charges > 0 or roll_cost == 0:
            return True
        if p.gold < roll_cost:
            return False
        # A refresh is only worth gold while a fresh shop can be expected to beat
        # the board's own floor. Measured against the best offer on the counter
        # instead — as this did — a T4 shop whose ceiling sits below the board
        # still looks worth +5, and whole wallets go into refreshing.
        if roll_benefit(p.tavern_tier, self._displaced_value(p, rl, rn)) <= _ROLL_FLOOR:
            return False

        if marginal is not None:
            need = max(1, self._replacement_cost(p))
            if p.gold >= need + roll_cost:
                cur = self._best_offer_value(p, rl, rn)
                after = (marginal + roll_benefit(p.tavern_tier, cur)) / (need + roll_cost)
                if after > marginal / need:
                    return True
            if marginal > 0:
                return False

        if p.gold - roll_cost >= self._replacement_cost(p):
            return True
        # The gold left cannot pay for a buy, but it does not survive the turn
        # either, and a refresh can still turn up a shop worth freezing.
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
