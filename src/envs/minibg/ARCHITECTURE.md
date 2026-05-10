# Mini BG 1v1 — module architecture

This module ships only **game rules** for Mini BG 1v1; no env / RL bindings.
It mirrors the layout of `src/envs/tictactoe/` (state + game) and adds card-game-specific layers.

## Files

- `effects.py` — `Keyword`, `Trigger`, frozen-dataclass effects (`SummonEffect`, `BuffRandomFriendly`, `StatAura`), and `Ability(trigger, effect)`.
- `state.py` — `Minion`, `PlayerState`, `MiniBGState` dataclasses.
- `cards.py` — `CARD_TEMPLATES` declarative pool, `make_minion(card_id)`, `shop_pool_for_tier(tier)`.
- `actions.py` — `Action` IntEnum (10 discrete shop actions) and gameplay constants.
- `battle.py` — pure battle simulator: `simulate_battle(p0_board, p1_board, p0_has_initiative, rng) -> (dmg_to_p0, dmg_to_p1)`.
- `game.py` — `MiniBGGame(TurnBasedGame[MiniBGState])`: shop loop, ON_BUY dispatch, round orchestration, battle invocation.

## Effect pattern

Static keywords (`Taunt`, `Shield`) live on `Minion.keywords: frozenset[Keyword]`. The battle code reads them directly during target selection and damage application — these keywords are not "abilities".

Triggered and continuous effects live on `Minion.abilities: tuple[Ability, ...]`. Each `Ability` carries:
- a `Trigger` (`ON_BUY`, `ON_DEATH`, `ON_TURN_END`, `AURA`),
- a typed `Effect` dataclass.

Dispatch happens in four places:

| Trigger        | Where                                | Resolver                                                    |
|----------------|--------------------------------------|-------------------------------------------------------------|
| `ON_BUY`       | `MiniBGGame._fire_on_buy`            | match on `effect` type, mutate buyer's board.               |
| `ON_DEATH`     | `battle._apply_on_death`             | match on `effect`; for `SummonEffect`, append to right end. |
| `ON_TURN_END`  | `MiniBGGame._fire_on_turn_end`       | left-to-right scan over finishing player's board.           |
| `AURA`         | `battle.attack_with_auras`           | sum `StatAura.attack` from other alive friendlies.          |

Adding a new card = one entry in `CARD_TEMPLATES` (data only).
Adding a new effect *type* = one frozen dataclass in `effects.py` + one branch in the matching dispatcher.

## Randomness

`MiniBGGame` owns a single `np.random.Generator`. `apply_action` is **not pure**: it consumes RNG for shop generation, Buffer's random target, battle target selection, and round-1 initiative.
This is by design for v0; if MCTS-style search is added later, RNG state can be pushed into `MiniBGState`.

## Round flow

Sequential shop: P0 shops fully (until `FINISH` or 10 BUY/SELL/ROLL/LEVEL_UP), then P1 shops, then battle resolves automatically inside the second player's terminating action.

`apply_action` returns the next state. Inside the second player's `FINISH` (or auto-finish on 10 actions), it:
1. Calls `simulate_battle` on deep copies of both boards.
2. Subtracts damage from each player's health.
3. Checks terminal conditions (any player ≤ 0 hp, or round 15 finished).
4. Otherwise increments `round_number`, restores gold (`gold_for_round`), auto-rerolls both shops, resets shop counters, sets `current_player_index = 0`.

## Battle copy semantics

`simulate_battle` builds `BattleMinion(template, current_health, shield_armed, deathrattle_fired)` from each `Minion`. The `template` is a reference to the actual board minion, but mutations only happen on `BattleMinion` fields. Permanent stats (`bonus_attack`, `bonus_health`) are read off the template via `Minion.raw_attack` and `Minion.max_health`. Aura bonuses are computed on demand at every attack read, so they vanish the moment the source minion dies.

After `simulate_battle` returns, `MiniBGGame` only applies `(dmg_to_p0, dmg_to_p1)` to player health. Player boards are never mutated by battle.

## Board ordering invariants

- Buy: appended to the rightmost free slot (`board.append(minion)`).
- Sell: list element removed at the chosen position; remaining minions shift left.
- Battle summons: appended to the rightmost end of the battle side, only if `alive_count < 4`.

## Action space

10 discrete actions (`Action` IntEnum). `legal_actions(state)` filters down to the currently valid subset:

- `BUY_SLOT_i` requires non-empty slot, board not full, gold ≥ 3, shop_actions_used < 10.
- `SELL_BOARD_i` requires position < len(board), shop_actions_used < 10.
- `ROLL` requires gold ≥ 1, shop_actions_used < 10.
- `LEVEL_UP` requires tier < 3 and gold ≥ cost(tier), shop_actions_used < 10.
- `FINISH` is always legal during shop phase.

After 10 BUY/SELL/ROLL/LEVEL_UP actions, only `FINISH` is legal; once consumed, `apply_action` auto-finishes the player.
