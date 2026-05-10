# Mini BG 1v1 — module architecture

Two layers:
- **Game core** (`actions.py`, `state.py`, `effects.py`, `cards.py`, `battle.py`, `game.py`) — pure rules.
- **RL wrapper** (`action_map.py`, `obs.py`, `env.py`) — fixed-size action space, self-centric vector observation, terminal-only reward.

## Files

- `effects.py` — `Keyword`, `Trigger`, frozen-dataclass effects (`SummonEffect`, `BuffRandomFriendly`, `StatAura`), and `Ability(trigger, effect)`.
- `state.py` — `Minion`, `PlayerState`, `MiniBGState` dataclasses.
- `cards.py` — `CARD_TEMPLATES` declarative pool, `make_minion(card_id)`, `shop_pool_for_tier(tier)`.
- `actions.py` — `Action` IntEnum (10 discrete shop actions) and gameplay constants.
- `battle.py` — pure battle simulator: `simulate_battle(p0_board, p1_board, p0_has_initiative, rng) -> (dmg_to_p0, dmg_to_p1)`.
- `game.py` — `MiniBGGame(TurnBasedGame[MiniBGState])`: shop loop, ON_BUY / ON_TURN_END dispatch, round orchestration, battle invocation, free `reorder_board` primitive.
- `action_map.py` — 33-action env layout: `ROLL` / `LEVEL_UP` / `BUY_SHOP_*` / `SELL_BOARD_*` / `SELECT_FINAL_ORDER_0..23`. Holds the precomputed permutation table and the env→game action mapper.
- `obs.py` — fixed-size vector observation (10 globals + 4·25 own board + 3·25 shop + 4·25 last-seen enemy board + 1 last-battle scalar).
- `env.py` — `MiniBGEnv(TurnBasedEnv)`: applies actions, fuses reorder + finish on `SELECT_FINAL_ORDER_*`, tracks last-seen enemy board and signed last-battle damage delta per player, emits self-centric obs. Read-only `state` / `game` for scripted opponents.
- `heuristic_bots/` — scripted opponents (`RandomBot`, `TempoBot`, …) + `tournament.run_tournament`. CLI: `python -m src.envs.minibg.heuristic_bots` or `python scripts/minibg_tournament.py`.

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

## Action space (game core)

10 discrete actions (`Action` IntEnum). `legal_actions(state)` filters down to the currently valid subset:

- `BUY_SLOT_i` requires non-empty slot, board not full, gold ≥ 3, shop_actions_used < 10.
- `SELL_BOARD_i` requires position < len(board), shop_actions_used < 10.
- `ROLL` requires gold ≥ 1, shop_actions_used < 10.
- `LEVEL_UP` requires tier < 3 and gold ≥ cost(tier), shop_actions_used < 10.
- `FINISH` is always legal during shop phase.

After 10 BUY/SELL/ROLL/LEVEL_UP actions, only `FINISH` is legal; once consumed, `apply_action` auto-finishes the player. `MiniBGGame.reorder_board(state, idx, perm)` is a free primitive (no action cost) used by the env to fuse reorder + finish.

## Action space (RL env)

33 discrete actions (`action_map.py`):

| Index range | Meaning                                                |
|-------------|--------------------------------------------------------|
| 0           | `ROLL`                                                 |
| 1           | `LEVEL_UP`                                             |
| 2..4        | `BUY_SHOP_0/1/2`                                       |
| 5..8        | `SELL_BOARD_0/1/2/3`                                   |
| 9..32       | `SELECT_FINAL_ORDER_0..23`: apply permutation **and** end shop phase. |

`PERMUTATIONS_4` is the 24-element list of permutations of `(0,1,2,3)` in lexicographic order (`itertools.permutations`); index 0 is the identity. For a board of size `k < 4`, only permutations whose tail beyond `k` is identity are legal (`legal_order_indices(k)`): `k=0` → 1, `k=1` → 1, `k=2` → 2, `k=3` → 6, `k=4` → 24. Empty board → only the identity `SELECT_FINAL_ORDER_0` ends the turn.

`MiniBGEnv.legal_actions_mask` projects `MiniBGGame.legal_actions(state)` onto this 33-slot vector and additionally enables the legal `SELECT_FINAL_ORDER_*` indices iff `FINISH` is legal in the game core.

## Observation (RL env)

Fixed-size float32 vector, **self-centric** (the current player is always "me"):

- 10 globals: `round/15`, `my_hp/15`, `enemy_hp/15`, `gold/8`, `gold_cap/8`, `my_tier/3`, `enemy_tier/3`, `actions_left/10`, `my_board_count/4`, `has_initiative_if_equal_board_size`.
- 4 × 25 own board slots, 3 × 25 shop slots, 4 × 25 last-seen enemy board slots. Each 25-D slot vector encodes: presence, card_id one-hot (10 cards), tier one-hot (3), 4 stat scalars (base/bonus attack/health), `Taunt`, `Shield`, runtime `has_shield`, and one flag per ability trigger (`ON_BUY`, `ON_DEATH`, `AURA`, `ON_TURN_END`).
- 1 last-battle scalar = `(damage_dealt − damage_taken) / 7` from the previous round, from this player's perspective.

The enemy's board is **only** updated post-battle (last-seen snapshot). The enemy's hp and tier are read live from current state — those are public.

## Reward (RL env)

Terminal-only: +1 win, −1 loss, 0 draw, all from the perspective of the player whose action just produced the terminal state. Illegal actions return reward `INVALID_ACTION_REWARD = -1.0` and do not mutate state.
