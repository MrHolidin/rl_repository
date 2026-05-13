# Mini BG 1v1 — game rules (v0.2)

This file documents the gameplay-level rules implemented by this module. Source of truth for ambiguities resolved during implementation.

## Setup
- Two players, both starting at 30 HP, 3 gold, Tavern Tier 1, empty board, freshly rolled shop (3 minions).
- At game start, one player is randomly chosen as the **odd-round initiative holder**. The other holds even-round initiative.

## Round structure
Each round has Shop Phase then Battle Phase. Game ends after a Battle that brings any player ≤ 0 HP, or after the 20th round.

### Shop Phase (sequential)
P0 shops first to completion, then P1. Implemented as one `current_player_index` pointer that flips when the current player finishes.

Available actions (max 20 BUY/SELL/ROLL/LEVEL_UP per round):

| Action      | Cost     | Effect                                                                 |
|-------------|----------|------------------------------------------------------------------------|
| Buy slot    | 3 gold   | Place shop[i] on right end of board; shop slot becomes empty; trigger ON_BUY abilities. |
| Sell pos    | +1 gold  | Remove minion from board; remaining minions shift left.                 |
| Roll        | 1 gold   | Replace **all 3 shop slots** with fresh minions of allowed tiers.       |
| Level up    | Variable price (base 5→7→8→11→11 for T1→…T6); **−1** each new round until you buy | Increase Tavern Tier (max 6). |
| Finish      | free     | End shop phase for this player.                                         |

After 20 actions, FINISH is forced automatically. FINISH itself is not counted in the 20.

**Board reordering** is free: the player can rearrange minions on their own board any number of times during their shop phase, at no cost and not counted against the 20-action limit. In the RL wrapper, reorder + finish is fused into a single `SELECT_FINAL_ORDER_*` action.

At round start, gold restores to round cap and **shops auto-reroll for free** for both players:

| Round | Gold cap |
|-------|----------|
| 1     | 3        |
| 2     | 4        |
| 3     | 5        |
| 4     | 6        |
| 5     | 7        |
| 6     | 8        |
| 7     | 9        |
| 8     | 10       |
| 9     | 10       |
| 10    | 10       |
| 11+   | 10       |

### Battle Phase
Resolved automatically at the moment both players have finished shopping. Operates on **copies** of player boards; permanent boards are not mutated.

#### First attacker
1. Side with strictly more alive minions attacks first.
2. Tie → side with this round's initiative goes first (odd rounds: initiative_player; even rounds: the other).

#### Attack loop
Sides alternate. Each side's "next attacker" is the next alive minion left-to-right after the previous one (wraps around to the leftmost alive).

#### Targeting
Target chosen uniformly at random among alive enemy minions. If at least one alive enemy has Taunt, target is uniformly random **only** among Taunt minions.

#### Damage exchange
Attacker and defender deal each other their attack value **simultaneously**. Then both sides' deaths are resolved.

#### Death resolution
For each side independently, dead minions trigger their `ON_DEATH` abilities once each, in left-to-right order (side 0 then side 1, list order within a side). Dead minions stay in the battle list as tombstones so indices remain stable. Summons append to the **right** end. **Combat board cap** is 7 living minions per side (`COMBAT_BOARD_MAX`); shop placement still uses 4 slots (`BOARD_SIZE`). When a summon would exceed 7 alive on that side, **extra minions are skipped** (nothing is destroyed to make room), matching BG summon overflow. **The Beast**–style effects summon on the **opponent** side but use the same cap and DR ordering relative to other queued `MinionDied` events.

#### Auras
Continuous stat auras recomputed on each strike read: **`StatAura`** (Raid Leader–style, never affects self), **`TribalOtherStatAura`** (your other minions of that tribe; `Race.ALL` matches any tribe), **`KeywordStatAura`** (minions with a given `Keyword`, e.g. Taunt), **`AdjacentStatAura`** (immediate board neighbors at index ±1 only). **During deathrattle / Kangor resolution**, attack and health contributions from these auras are **off** (same `death_resolution` flag as the implementation). **Health** from auras is tracked in combat via snapshot resync so when a Mal'Ganis-like source dies, other demons lose bonus current HP up to the new cap.

**Combat adjacency** matches BG with tombstones: dead minions stay in the list, so “adjacent” is strictly ±1 index, not nearest living.

**Defender of Argus** (shop): `BuffAdjacentBattlecry` applies +1/+1 and Taunt to left/right neighbors in **persistent** board order when played.

#### Shield
`Keyword.SHIELD` minions enter battle with `shield_armed=True`. The first incoming damage is fully absorbed and `shield_armed` flips to False. Shield re-arms at the start of every battle.

#### Battle end
When one or both sides have no alive minions left.

#### Damage to losing player
`damage = min(7, 1 + sum(tier of surviving winner's minions))`. Battle draw (both sides empty) deals 0 damage. The constant +1 means even a single Tier 1 survivor inflicts 2 damage.

## Persistence between rounds
Permanent (kept):
- Player health, gold cap, tavern tier.
- **Combat survivors** on each player's board (composition and `bonus_*` buffs on copied templates; see below).
- Hand cards (including stats buffed by `ON_TURN_START` in hand).
- Permanent stat buffs from shop (`ON_PLACE`, `ON_BUY`, `ON_TURN_END`, `ON_TURN_START`, etc.).

After each battle, the shop board is replaced by **alive** combat minions (shallow copy of each survivor's template, shields re-armed if the minion has Divine Shield). Combat damage to minions is not kept on the persistent minion (templates restore to printed + permanent `bonus_*`). **Summoned tokens** and Khadgar-multiplied summons persist when they survive the brawl. If more than `BOARD_SIZE` minions survive, only the first `BOARD_SIZE` in combat order are kept.

Transient (combat-only for a single brawl):
- In-combat current HP / shield-armed runtime on `BattleMinion` (not stored on shop `Minion`).
- Corpses and minions that die before battle ends.

## Card pool

| Card           | Tier | Stats | Keyword(s) | Ability                                              |
|----------------|------|-------|------------|------------------------------------------------------|
| Recruit        | 1    | 2/2   | —          | —                                                    |
| Guard          | 1    | 1/3   | Taunt      | —                                                    |
| Buffer         | 1    | 1/1   | —          | ON_BUY: +1/+1 to a random other friendly minion.     |
| Bruiser        | 2    | 4/3   | —          | —                                                    |
| Shield Bot     | 2    | 2/2   | Shield     | —                                                    |
| Pack Rat       | 2    | 2/2   | —          | ON_DEATH: summon `rat_token` (1/1).                  |
| Big Guy        | 3    | 5/5   | —          | —                                                    |
| Commander      | 3    | 3/4   | —          | AURA: other friendly minions get +1 attack.          |
| Summoner       | 3    | 4/3   | —          | ON_DEATH: summon `summoned_token` (2/2).             |
| Micro Machine | 1    | 2/1   | Mech       | `ON_TURN_START` (after round increment, before shop reroll): +1 Attack (permanent). |
| Mentor         | 3    | 1/3   | —          | ON_TURN_END: +2/+1 to a random other friendly minion (permanent). |
| rat_token      | 1    | 1/1   | —          | token, never appears in shop.                        |
| summoned_token | 1    | 2/2   | —          | token, never appears in shop.                        |

Tavern tier T allows shop minions of tier ≤ T. Pool is treated as infinite (sample with replacement).

## Buffer specifics
- "Other random friendly" excludes the just-bought Buffer itself.
- If no other friendlies exist, the effect is a no-op.
- Buff stacks: each Buffer purchase grants +1/+1 to one (chosen-uniformly) existing friendly.

## Termination
- Either player ≤ 0 HP after a battle: that player loses (the other wins).
- Both players ≤ 0 HP after the same battle: draw (winner = 0). (Unreachable with the v0 mechanics — only the loser of a battle takes damage — but kept defensively.)
- Round 15 ends with both alive: draw (winner = 0).
