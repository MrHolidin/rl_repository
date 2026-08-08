# Ability encoding: plan

Two stages, no intermediate. **Stage 1** replaces the effect id with a semantic
decomposition. **Stage 2** lifts abilities out of the per-slot summary and into
the attention sequence as their own tokens.

Stage 2 depends on stage 1: attention over id-shaped tokens just learns pairwise
combinations of unique identifiers, which is what already happens, only more
expensively. Axes are what make two ability tokens comparable, and only then
does attention between them transfer anything.

Everything below is measured on patch `19_6_0_74257` unless stated.

---

## Why

**The effect id is nearly a card id.** 73 registered effect classes, 66 appear
in the patch, and **71% of those appear on exactly one card**. So the "effect"
channel does not generalise: learning `Ghastcoiler`'s effect transfers nowhere.
It is an alias for identity, and 62% of the observation is built on it.

That shows up in the swap probe (attn3 final, 8831 shop decisions, donor of the
same tier, measured as the change in the policy's p(buy that slot); base p =
0.0693):

| channel swapped | mean \|Δp\| |
|---|---|
| tier (donor of a *different* tier) | 0.0857 |
| keywords + shield + golden | 0.0456 |
| ability block | 0.0338 |
| race | 0.0278 |
| card_idx | 0.0237 |
| stats (atk/hp) | 0.0105 |
| whole card | 0.1105 |

Ability tokens and card_idx sit at 0.0338 and 0.0237 — close, because they carry
nearly the same information. Tier is the strongest single channel and moves
directionally (relabelled up: +0.0969 over 6432 cases; down: −0.0412 over 2399),
i.e. the policy uses it as a power proxy.

**The ability block is mostly padding.** `K_ABIL = 4` slots per minion, but the
patch has at most **2** abilities on any card (0 on 39 cards, 1 on 116, 2 on 4).
Nothing is ever truncated. In live games **10.4%** of ability tokens are
non-empty: of the block's 1560 floats, ~162 carry information and ~1397 are
zeros — 55% of the whole observation.

**Grouping is what buys transfer.** A first-pass taxonomy covers all 73 classes:

| KIND | classes | card-abilities |
|---|---|---|
| BUFF_STATS | 36 | 67 |
| SUMMON | 6 | 23 |
| DAMAGE | 7 | 10 |
| ADD_CARD | 6 | 6 |
| GRANT_KEYWORD | 3 | 4 |
| ECONOMY | 4 | 4 |
| MULTIPLIER | 3 | 3 |
| COMBAT_RULE | 3 | 3 |
| TRANSFORM | 3 | 2 |
| META | 2 | 2 |

36 singleton-ish classes collapse into one group covering 67 card-abilities.

---

## Stage 1 — semantic encoding

### Axes

Added to the ability token, alongside the existing trigger / condition / tribe /
keyword / summon-token fields.

| axis | values | notes |
|---|---|---|
| `kind` | 10, **multi-hot** | see table above |
| `scope` | ~15, one-hot | who it hits |
| `scaling` | 4, one-hot | flat / per-board-count / per-own-stat / multiplicative |
| `persistence` | 2 | aura (dies with the host) vs one-shot |

`kind` must be **multi-hot, not one-hot**: `ConsumeFriendlyBattlecry` destroys a
friendly minion, buffs self by a multiplier *and* grants gold;
`SummonRandomAndCopyToHandEffect` summons *and* adds a card.

`scope` must include **positional** targets, not just relational ones —
`BuffLeftmostRepeatedEffect` (Majordomo) and `DealDamageLeftmostEnemyMinion` hit
the leftmost slot, and attack order is left-to-right, so position is load-bearing
and couples to the ordering head. It must also distinguish **"the summoned
minion"** from **"the listener"**: `BuffSummonedIfRace` buffs what was just
summoned, `BuffListenerIfSummonedMatches` buffs *self* when something matching is
summoned. Same trigger, same tribe filter, opposite target.

`scaling` needs the fourth value for `SummonEffect.count_from_source_attack`,
which scales with the source's own attack — neither a board count nor a
multiplier.

`persistence` is what separates `StatAura` from `BuffAllFriendlyMinions`: the
first evaporates when the host dies, the second is permanent. Currently
indistinguishable.

### Numeric fields

`encode_ability_token` reads exactly `attack`, `health`, `amount`, `repeats`,
`count`; `_numeric_param` returns 0 for anything else. 18 of 73 effect classes
name their fields differently.

**Most of that is harmless** — those fields are constant across the patch:
`attack_per`, `health_per`, `amount_per_match`, `stat_multiplier`, `per_attack`
are all `1`. Knowing an effect is per-count is enough when the per-unit value is
always one.

**What genuinely goes missing: 9 effect classes, 10 of 124 abilities (8%).** The
one that matters is `factor` on the three multiplier auras — **Brann Bronzebeard**
and **Baron Rivendare** are t5 first-tier cards whose ×2-or-×3 is invisible to the
policy. Add a `factor` channel.

### Fields to drop

Constant across the patch, so they encode nothing:
`for_opponent` (never `True`), `make_golden`, `copy_golden`, `both_adjacent`
(all `False`), `freeze_slot` (always `True`), `dr_wave_count` (always `1`).

There is **no `side` axis** — no card in this patch summons for the opponent.

Flags that do vary, and should be encoded: `exclude_self`, `grant_taunt`,
`legendary_only`, `require_deathrattle`, `attack_immediately`,
`count_from_source_attack`, `exact_tier`.

### K_ABIL 4 → 2

Free: nothing is truncated (patch max is 2), the block drops 1560 → 780 and the
observation ~2683 → ~1900 (−29%). Collect is CPU-bound and forward is 72% of it,
so this is real throughput. Guard with a catalog assert at patch load, in the
same style as the existing `_EFFECT_CLASSES` completeness check.

This disappears in stage 2 (tokens are emitted per real ability with a mask, so
padding is gone by construction). It is worth doing in stage 1 because stage 1
gets trained on its own.

### Where the code goes

- **New** `src/envs/bglike/effect_taxonomy.py` — the class → (kind, scope,
  scaling, persistence) table plus an **import-time assert that every registered
  effect has a row**. Mirrors the `_EFFECT_CLASSES` guard in `minibg/obs.py`; a
  new effect then cannot be added silently.
- `src/envs/bglike/obs_v5.py` — `encode_ability_token` gains the new fields;
  `ABIL_FEAT_DIM` changes.
- The net's `AbilityTokenEncoder` — embeddings for the four new categorical axes.
- Config flag `effect_encoding: id | semantic | both`. Default `id` is bit-exact
  current behaviour. `both` keeps the raw id as a residual channel; `semantic`
  drops it, and only `semantic` answers whether mechanics suffice without an
  identity shortcut.

`ABIL_FEAT_DIM` changing means the observation dimension changes, so this is a
**new obs/network version and a from-scratch retrain**, not an in-place edit.
Budget for the full registration checklist: registering a `bglike_structured_vN`
takes ~6 enumeration edits beyond the model and factory, and a missing
`use_structured` tuple in `run_distributed` drops training into the flat path and
crashes on `legal_mask`.

---

## Stage 2 — abilities as attention tokens

`_ability_summary` currently pools a minion's abilities into a per-slot vector
**before** the entity attention. Abilities of different minions therefore never
see each other, while synergy is exactly cross-minion ability interaction —
Kangor's Apprentice pulls mechs, Amalgadon counts tribes.

Emit each real ability as its own token in the attention sequence, with a host
back-reference (which slot it belongs to) and a mask. At 10.4% occupancy that is
roughly 6 extra tokens per lobby state — cheap.

Do this only after stage 1 is trained and measured. With id-shaped tokens,
attention has nothing to generalise over.

---

## Validation

**Before any training** — reuse the swap probe. If the axes carry meaning, a swap
to a donor with the **same `kind` and `scope`** must move p(buy) measurably less
than a swap to a donor with a **different `kind`**. Under the current encoding no
such structure exists and the two should be indistinguishable. This tests the
representation without a single training step.

**Import-time** — the taxonomy completeness assert.

**After training** — a paired per-lobby head-to-head against a base run at
matched steps. Per-lobby pairing is the only correct statistic here: placements
in a lobby sum to 36, so the two team means always sum to 9.0 and are perfectly
anti-correlated; independent CIs overstate the uncertainty. 150 lobbies gives
SE ≈ 0.13 places.

**Do not** read self-play league ratings across runs — each run's pool is its own
frame of reference.

---

## Order and dependency

Run on top of the `no-card-emb` branch. While the card-identity embedding is
available, the policy can keep playing from memory and no result about mechanics
is interpretable — the ablation is the gate. First evidence is good: at 4.58M of
5M the ablated run shows a monotone frozen ladder (46.13 → 51.27 across slots
7–21) with the learner mid-pool, i.e. it learns without card identity at all.

Keep stage 1 and stage 2 as separate runs even though each costs a retrain. Every
combined intervention this project has tried — DvD, RND, tier shaping, the
entropy controller, and shaping+entropy together — moved behaviour without moving
strength, and the combined one could not be attributed afterwards.

---

## Retracted along the way

Recorded so they are not re-proposed:

- **"Drop the tier one-hot, it measured 0.0000."** Wrong: the probe used
  same-tier donors, so the channel was never perturbed. Measured properly it is
  the *strongest* channel at 0.0857.
- **"Add a `side` axis for `for_opponent`."** No card in the patch sets it.
- **"25% of abilities lose their numbers."** Overcounted from field names; 14 of
  those fields are patch-constants. The real figure is 8%.
