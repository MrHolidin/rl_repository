# Mini BG 1v1 — game rules (v0.2)

This file documents the gameplay-level rules implemented by this module. Source of truth for ambiguities resolved during implementation.

## Setup
- Two players, both starting at 15 HP, 3 gold, Tavern Tier 1, empty board, freshly rolled shop (3 minions).
- At game start, one player is randomly chosen as the **odd-round initiative holder**. The other holds even-round initiative.

## Round structure
Each round has Shop Phase then Battle Phase. Game ends after a Battle that brings any player ≤ 0 HP, or after the 15th round.

### Shop Phase (sequential)
P0 shops first to completion, then P1. Implemented as one `current_player_index` pointer that flips when the current player finishes.

Available actions (max 10 BUY/SELL/ROLL/LEVEL_UP per round):

| Action      | Cost     | Effect                                                                 |
|-------------|----------|------------------------------------------------------------------------|
| Buy slot    | 3 gold   | Place shop[i] on right end of board; shop slot becomes empty; trigger ON_BUY abilities. |
| Sell pos    | +1 gold  | Remove minion from board; remaining minions shift left.                 |
| Roll        | 1 gold   | Replace **all 3 shop slots** with fresh minions of allowed tiers.       |
| Level up    | 4 (T1→T2), 6 (T2→T3) | Increase Tavern Tier (max 3).                              |
| Finish      | free     | End shop phase for this player.                                         |

After 10 actions, FINISH is forced automatically. FINISH itself is not counted in the 10.

At round start, gold restores to round cap and **shops auto-reroll for free** for both players:

| Round | Gold cap |
|-------|----------|
| 1     | 3        |
| 2     | 4        |
| 3     | 5        |
| 4     | 6        |
| 5     | 7        |
| 6+    | 8        |

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
For each side independently, dead minions trigger their `ON_DEATH` abilities once each, in left-to-right order. Dead minions stay in the battle list as tombstones (skipped for further targeting / attacking) so attack-order indices remain stable. Summons append to the right end if alive count < 4.

#### Auras
Auras are **continuous**. `attack_with_auras(minion)` recomputes a minion's effective attack on every read by summing `StatAura.attack` from all other alive friendlies. The instant the source dies, the bonus disappears.

#### Shield
`Keyword.SHIELD` minions enter battle with `shield_armed=True`. The first incoming damage is fully absorbed and `shield_armed` flips to False. Shield re-arms at the start of every battle.

#### Battle end
When one or both sides have no alive minions left.

#### Damage to losing player
`damage = min(7, 1 + sum(tier of surviving winner's minions))`. Battle draw (both sides empty) deals 0 damage. The constant +1 means even a single Tier 1 survivor inflicts 2 damage.

## Persistence between rounds
Permanent (kept):
- Player health, gold cap, tavern tier.
- Bought / sold minions on the permanent board.
- Permanent stat buffs (e.g., Buffer's +1/+1 stacked on `bonus_attack` / `bonus_health`).

Transient (battle-only, reset each Battle Phase):
- Damage taken in battle.
- Shield armed/spent state.
- Summoned tokens (rat_token, summoned_token).
- Death.

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
