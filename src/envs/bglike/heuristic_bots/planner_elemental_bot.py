"""Elemental (Nomi) line built on the planner, as a profile rather than a rewrite.

Nomi, Kitchen Nightmare (BGS_104, tier 5) gives Elementals in the tavern +1/+1
after every Elemental *played* — permanently, and to every future offer, since
the engine keeps the running total in ``shop_elemental_bonus`` and applies it to
each slot it fills. So the asset of this line is the counter, not the board: a
minion that is bought, played and sold has already paid, and the elementals you
buy later come out of the tavern several times their printed size.

That makes the line expressible in the planner's own currency, without the
card-id tables and "after Nomi only buy elementals" filter the standalone
``elemental`` bot needed:

* the accumulated bonus is already on the shop minions themselves, so the body
  half of the value needs no change at all;
* the play half — one tick of the counter — is a stream over the elementals
  still to be bought, exactly the shape ``_level_roi`` already uses for a
  tier-up, and it is credited only to a minion that has not been played yet,
  which is what makes selling a spent one cheap and lets the cycle run;
* what those two terms do together is a ramp, not a mill: early on the bodies
  are small and the stream is long, so cheap elementals are worth cycling; as
  the counter grows and the rounds run out the bodies win and the same knapsack
  starts keeping them. No phase switch is written down anywhere.

Below tier 5 this is the plain planner. At tier 5 it hunts: a fresh shop shows
Nomi about 1.5% of the time with a random tribe excluded and 3.7% with murlocs
out (scratchpad calibration, 20k shops), which is worth far more per gold than
any ordinary refresh — while still leaving enough to replace a minion, so the
board does not rot during the search.
"""

from __future__ import annotations

from typing import Optional

from src.bg_recruitment.economy import effective_buy_cost, effective_roll_cost
from src.bg_core.effects import (
    AddRandomMinionToHandEffect,
    AddTokenToHandEffect,
    Trigger,
)
from src.bg_core.minion import Minion, Race
from src.bg_lobby.player import PlayerState

from ..action_map import A_PLACE_BASE, A_SELL_BASE
from ..actions import HAND_SIZE
from src.envs.minibg.heuristic_bots.value_model import rounds_left_estimate

from .planner_bot import SRC_BOARD, PlannerHeuristicBot
from .bots import HeuristicEnv, _mask, _me

# Nomi and his golden copy.
NOMI_IDS = frozenset({"BGS_104", "TB_BaconUps_201"})

# Tier by round. Fixed on purpose: the ROI rule the planner uses reaches tier 4
# by round 5 on a three-minion board, and this line cannot afford that — it has
# to still be alive when the counter starts paying.
_TIER_SCHEDULE = ((3, 2), (5, 3), (7, 4), (9, 5))
# The sixth tier is not on the schedule: Nomi lives on the fifth, and forcing the
# upgrade takes eight of the ten gold on exactly the turns the counter should be
# compounding — three of the four wallet-full turns that produced no tick at all
# opened with it. It is taken only out of gold the cycle does not need.
_TIER_SIX_ROUND = 12
_TIER_SIX_SPARE = 6

# Value of one +1/+1 on an elemental yet to be bought, in the planner's units:
# 1 + 1 + 0.055*(a+h+1) for the stat lines this line actually buys.
_TICK_STAT_VALUE = 2.5
# Elementals bought per turn once the cycle is running.
_ELEM_BUYS_PER_TURN = 3.0


def _has_nomi(p: PlayerState) -> bool:
    return any(m.card_id in NOMI_IDS for m in p.board)


def _is_elemental(m) -> bool:
    # m may be a TavernSpell (no race attribute) if it's sitting in hand.
    return getattr(m, "race", None) == Race.ELEMENTAL


class PlannerElementalBot(PlannerHeuristicBot):
    name = "planner_elemental"

    # ------------------------------------------------------------------
    # Fixed curve
    # ------------------------------------------------------------------

    def _target_tier(self, p: PlayerState, rn: int) -> int:
        target = 1
        for round_no, tier in _TIER_SCHEDULE:
            if rn >= round_no:
                target = tier
        return target

    def _should_level_up(self, env: HeuristicEnv, p: PlayerState, rl: int) -> bool:
        rn = env.state.round_number
        cost = p.next_tier_up_cost
        if p.gold < cost:
            return False
        if p.tavern_tier < self._target_tier(p, rn):
            return True
        # Above the schedule, only with gold the cycle will not miss.
        return (
            p.tavern_tier == 5
            and rn >= _TIER_SIX_ROUND
            and p.gold - cost >= _TIER_SIX_SPARE
        )

    # ------------------------------------------------------------------
    # Hunting for Nomi
    # ------------------------------------------------------------------

    def _searching_for(self, p: PlayerState):
        """What a refresh is being spent on, or ``None`` if it should not be.

        Before Nomi that is Nomi himself; after him it is any Elemental, since
        each one played is a tick of the counter. Either way the refresh check
        runs *ahead* of the buy, so it has to stand down once the thing it is
        looking for is on the counter — otherwise the search rolls away exactly
        what it was searching for. That mistake cost this bot both halves of the
        line: Nomi appeared in these shops 75 times over ten lobbies and reached
        a board five frames in total, and once fixed for Nomi but not for the
        cycle, the counter fell from 2.3 to 0.6 by round 15.
        """
        if not _has_nomi(p):
            if p.tavern_tier < 5:
                return None
            want = lambda m: m.card_id in NOMI_IDS  # noqa: E731
        else:
            want = _is_elemental
        if any(m is not None and want(m) for m in p.hand):
            return None
        # Standing down for a copy that cannot be paid for stalls the turn
        # outright: the search stops and the buy never happens. Sixteen of
        # sixty-nine post-Nomi turns ended with a full wallet and no tick.
        if p.gold >= effective_buy_cost(p) and any(
            m is not None and want(m) for m in p.shop
        ):
            return None
        return want

    def _should_roll(
        self,
        env: HeuristicEnv,
        p: PlayerState,
        mask,
        legal,
        rl: int,
        rn: int,
        marginal: Optional[float],
    ) -> bool:
        # Before Nomi a refresh is worth P(Nomi) times a stream that dwarfs any
        # single card. After him it is worth the chance of another Elemental to
        # play, which is a tick of the counter — a shop of five shows one about
        # every other refresh, and one tick outweighs anything the ordinary bar
        # is comparing. Without this channel the cycle has no fuel: Elementals
        # arrive on the counter roughly once a turn on their own, and the
        # counter crawls (1.3 by round 12) instead of compounding.
        if self._searching_for(p) is not None and self._tick_value(rl) > 0.0:
            from ..action_map import A_ROLL

            if bool(mask[A_ROLL]) and A_ROLL in legal:
                roll_cost = effective_roll_cost(p)
                if p.free_roll_charges > 0 or roll_cost == 0:
                    return True
                # Hold back only enough to still buy what the refresh turns up.
                if p.gold - roll_cost >= self._replacement_cost(p):
                    return True
        return super()._should_roll(env, p, mask, legal, rl, rn, marginal)

    # ------------------------------------------------------------------
    # The counter, priced like any other stream
    # ------------------------------------------------------------------

    def _tick_value(self, rl: int) -> float:
        """What one more play of an Elemental adds to every purchase still to come."""
        return _TICK_STAT_VALUE * _ELEM_BUYS_PER_TURN * float(max(0, rl))

    def _unlock_value(self, rl: int) -> float:
        """What owning Nomi is worth: every tick of the counter, for the rest of the run.

        Without this the line hunts a card it then declines to buy. Nomi is a 4/4
        with a trigger, which the board value puts at 15-18 against a board of
        25-40, so he never makes the target set — and the tick stream only starts
        being credited once he is *already* owned. Over ten lobbies he turned up
        on the counter 75 times and reached the board five frames in total.

        The stream is the sum over the plays still to come of what each adds to
        the purchases after it: ``n`` plays, each worth a tick to the ones that
        follow, is ``n^2 / 2`` stat-pairs.
        """
        plays = _ELEM_BUYS_PER_TURN * float(max(0, rl))
        return _TICK_STAT_VALUE * plays * plays / 2.0

    def _extra_tick_value(self, m: Minion, rl: int, *, played: bool) -> float:
        """Cards that hand you another Elemental are worth another tick.

        The rate of the counter is set by how many Elementals you can get hold
        of in a turn, and the shop only shows about one — a tick costs three
        gold to buy, one comes back on the sale, and roughly two more go on
        refreshes hunting the next one, which caps the cycle at two or three
        ticks on a ten gold turn. Two cards break that cap and neither was
        priced: Sellemental hands over a 2/2 Elemental token when *sold*, and
        Tavern Tempest's battlecry adds one to hand. The stock value model
        charges +2.5 and +2.2 for those against a tick worth around 45.

        The token arrives on the sale, so for a Sellemental already on the board
        the tick is what holding it gives up — a negative, which puts it first
        in line to be sold, which is exactly how the token is collected.
        """
        total = 0.0
        tick = self._tick_value(rl)
        for ab in m.abilities:
            eff = ab.effect
            if ab.trigger == Trigger.ON_SELL and isinstance(eff, AddTokenToHandEffect):
                if self._token_is_elemental(eff.token_id):
                    total += -tick * float(eff.count) if played else tick * float(eff.count)
            elif (
                not played
                and ab.trigger == Trigger.ON_PLACE
                and isinstance(eff, AddRandomMinionToHandEffect)
                and eff.tribe == Race.ELEMENTAL
            ):
                total += tick
        return total

    def _token_is_elemental(self, token_id: str) -> bool:
        patch = getattr(self, "_patch", None)
        if patch is None:
            return False
        tpl = patch.templates.get(token_id)
        return tpl is not None and tpl.race == Race.ELEMENTAL

    def _cycle_step(
        self, mask, p: PlayerState, rl: int, rn: int
    ) -> Optional[int]:
        """Play an Elemental for the tick even when it will be sold straight back.

        The knapsack compares *end-of-turn boards*, so a play that is undone is
        invisible to it — which meant the counter only ever moved when an
        Elemental was good enough to keep, and Elementals are only good once the
        counter is big. The cycle is the way out of that circle: sell the spent
        Elemental from the last turn, play the new one, and the board ends where
        it started two gold poorer and one tick richer.
        """
        if not _has_nomi(p):
            return None
        tick = self._tick_value(rl)
        if tick <= 0.0:
            return None
        held = [
            (i, m)
            for i, m in enumerate(p.hand[: min(HAND_SIZE, len(p.hand))])
            if m is not None and _is_elemental(m)
        ]
        if not held:
            return None
        i, m = held[0]
        if bool(mask[A_PLACE_BASE + i]):
            return A_PLACE_BASE + i

        # Board is full: free the slot, preferring a minion whose own play value
        # is already spent — in the steady cycle that is last turn's Elemental,
        # so nothing that matters is given up.
        cands = self._candidates(p, rl, rn)
        board = [
            c
            for c in cands
            if c.src == SRC_BOARD and c.m.card_id not in NOMI_IDS
            and bool(mask[A_SELL_BASE + c.idx])
        ]
        if not board:
            return None
        worst = min(board, key=lambda c: c.v)
        body = self._v(m, board_len=len(p.board), dominant=None, rl=rl, rn=rn)
        if tick + body <= worst.v:
            return None
        return A_SELL_BASE + worst.idx

    def _choose_action(self, env: HeuristicEnv) -> int:
        mask = _mask(env)
        p = _me(env)
        if p.pending_choice is None and getattr(env, "rl_pending", None) is None:
            rn = env.state.round_number
            step = self._cycle_step(mask, p, rounds_left_estimate(rn), rn)
            if step is not None:
                return int(step)
        return super()._choose_action(env)

    def _play_value_correction(
        self, m: Minion, p: PlayerState, rl: int, rn: int, *, played: bool
    ) -> float:
        base = super()._play_value_correction(m, p, rl, rn, played=played)
        if _has_nomi(p):
            base += self._extra_tick_value(m, rl, played=played)
        if m.card_id in NOMI_IDS:
            # Priced on both sides of the board line on purpose: the stream runs
            # while he is alive, so it is equally a reason to buy him and a
            # reason never to sell him.
            return base + self._unlock_value(rl)
        if played or not _is_elemental(m) or not _has_nomi(p):
            return base
        return base + self._tick_value(rl)


__all__ = ["PlannerElementalBot"]
