# Mini BG 1v1 — module architecture

Two layers:
- **Game core** (`actions.py`, `state.py`, `effects.py`, `cards.py`, `battle.py`, `game.py`) — pure rules.
- **RL wrapper** (`action_map.py`, `obs.py`, `env.py`) — fixed-size action space, self-centric vector observation, terminal-only reward.

## Files

- `effects.py` — `Keyword`, `Trigger`, frozen-dataclass effects (`SummonEffect`, `BuffRandomFriendly`, `StatAura`), and `Ability(trigger, effect)`.
- `state.py` — `Minion`, `PlayerState` (with `board`, `shop`, `hand: List[Optional[Minion]]`, `phase: PlayerPhase`, `shop_actions_used`), `MiniBGState`. `PlayerPhase` is `SHOP | ORDER | DONE`.
- `cards.py` — `CARD_TEMPLATES` declarative pool, `make_minion(card_id)`, `shop_pool_for_tier(tier)`.
- `actions.py` — `Action` IntEnum (13 discrete game actions: BUY×3, SELL×4, ROLL, LEVEL_UP, FINISH, PLACE_HAND×3) and gameplay constants (`SHOP_SIZE=3`, `BOARD_SIZE=4`, `HAND_SIZE=3`).
- `battle.py` — pure battle simulator: `simulate_battle(p0_board, p1_board, p0_has_initiative, rng) -> (dmg_to_p0, dmg_to_p1)`.
- `game.py` — `MiniBGGame(TurnBasedGame[MiniBGState])`: shop loop, hand mechanic, two-phase shop turn (`shop` → `order`), ON_BUY / ON_TURN_END dispatch, round orchestration, battle invocation, `reorder_board` primitive (compact-after-permute).
- `action_map.py` — 37-action env layout: `ROLL` / `LEVEL_UP` / `BUY_SLOT_*` / `SELL_BOARD_*` / `PLACE_HAND_*` / `FINISH` / `SELECT_ORDER_0..23`. Holds the precomputed permutation table and the env→game action mapper.
- `obs.py` — fixed-size vector observation (10 globals + 4·25 own board + 3·25 shop + 3·25 hand + 4·25 last-seen enemy board + 1 last-battle scalar + 1 phase indicator).
- `env.py` — `MiniBGEnv(TurnBasedEnv)`: applies actions, on `SELECT_ORDER_*` calls `reorder_board(perm)` then submits the order via `apply_action(FINISH)` from the order phase. Tracks last-seen enemy board and signed last-battle damage delta per player, emits self-centric obs. On battle resolution emits `info["battle_signed"] = (signed_p0, signed_p1)` for symmetric per-player shaping; the env's own `step().reward` carries only terminal win/loss/draw (shaping is computed in `src.training.agent_perspective_env.AgentPerspectiveEnv`). Read-only `state` / `game` for scripted opponents. Optional `replay_path` / `replay_meta`: JSONL per-step snapshots via `replay.py` (replay format `2` includes `hand` and `phase`). Training: the trainer consumes `AgentPerspectiveEnv` which wraps `MiniBGEnv`, drives opponent moves until the agent must act again, and attributes terminal + shaping reward in the agent's zero-sum perspective.
- `heuristic_bots/` — scripted opponents (`RandomBot`, `TempoBot`, …) + `tournament.run_tournament`. CLI: `python -m src.envs.minibg.heuristic_bots` or `python scripts/minibg_tournament.py`. The shared `_finish(env)` helper is phase-aware: in shop phase it auto-places pending hand cards then issues `FINISH`; in order phase it picks a permutation via `choose_final_order`.
- **Shared eval / matches** — `MiniBGEnv` follows the same `TurnBasedEnv` contract as Connect4/Othello (`legal_actions_mask`, `current_player_token`, `winner` in `±1` / `0`), so `src.utils.match.play_single_game` / `play_match` work with multi-step shop turns (each loop picks the actor whose `current_player_token` matches `agent_token`). Use `play_match(..., game_id="minibg")` when no custom `env` is passed, or `make_game("minibg", ...)`. Heuristic bots implement `choose_action(env)`; for `play_match` use `MiniBGHeuristicAgent` from `heuristic_bots.agent_adapter` (set via `set_env`). `src.evaluation.eval_checkpoints.build_opponents_from_names(..., game_id="minibg")` accepts keys from `default_bot_constructors()` (e.g. `tempo`, `buffer_t2`, `random`).

## Effect pattern

Static keywords (`Taunt`, `Shield`) live on `Minion.keywords: frozenset[Keyword]`. The battle code reads them directly during target selection and damage application — these keywords are not "abilities".

Triggered and continuous effects live on `Minion.abilities: tuple[Ability, ...]`. Each `Ability` carries:
- a `Trigger` (`ON_BUY`, `ON_DEATH`, `ON_TURN_END`, `AURA`),
- a typed `Effect` dataclass.

Dispatch happens in four places:

| Trigger        | Where                                | Resolver                                                    |
|----------------|--------------------------------------|-------------------------------------------------------------|
| `ON_BUY`       | `MiniBGGame._fire_on_buy`            | match on `effect` type; `BuffRandomFriendly` picks a random target from `board` only. The bought minion lands in hand at trigger time, so empty board → no-op. |
| `ON_DEATH`     | `battle._apply_on_death`             | match on `effect`; for `SummonEffect`, append to right end. |
| `ON_TURN_END`  | `MiniBGGame._fire_on_turn_end`       | left-to-right scan over the player's board at `submit_order` time (the moment `phase` flips to `DONE`). |
| `AURA`         | `battle.attack_with_auras`           | sum `StatAura.attack` from other alive friendlies.          |

Adding a new card = one entry in `CARD_TEMPLATES` (data only).
Adding a new effect *type* = one frozen dataclass in `effects.py` + one branch in the matching dispatcher.

## Randomness

`MiniBGGame` owns a single `np.random.Generator`. `apply_action` is **not pure**: it consumes RNG for shop generation, Buffer's random target, battle target selection, and round-1 initiative.
This is by design for v0; if MCTS-style search is added later, RNG state can be pushed into `MiniBGState`.

## Round flow

Each round is split into a shop phase and an order phase per player.

**Shop phase** (`PlayerPhase.SHOP`): the active player issues any sequence of `BUY` / `SELL` / `PLACE` / `ROLL` / `LEVEL_UP` actions; each costs one of `MAX_SHOP_ACTIONS` budget slots. None of these actions pass the turn. `FINISH` ends the shop phase by flipping `phase` → `ORDER` (also without passing the turn). If the action budget hits the cap, `apply_action` auto-flips to `ORDER`.

**Order phase** (`PlayerPhase.ORDER`): only the env-level `SELECT_ORDER_j` action is legal. The env layer applies `reorder_board(state, idx, perm)` and then calls `apply_action(FINISH)`; from order phase, `FINISH`:
1. Sets `phase = DONE`.
2. Fires `ON_TURN_END` triggers on the now-finalized board.
3. Either passes the turn to the other player (if they are not yet `DONE`) or resolves the battle.

**Battle resolution** (`_resolve_battle_and_advance`):
1. `simulate_battle` on deep copies of both boards.
2. Subtract damage from each player's health.
3. Terminal check (any player ≤ 0 hp, or round 15 finished).
4. Otherwise increment `round_number`, restore gold (`gold_for_round`), reroll both shops, clear `shop_actions_used`, set both phases back to `SHOP`. **Hand persists across rounds.**

## Battle copy semantics

`simulate_battle` builds `BattleMinion(template, current_health, shield_armed, deathrattle_fired)` from each `Minion`. The `template` is a reference to the actual board minion, but mutations only happen on `BattleMinion` fields. Permanent stats (`bonus_attack`, `bonus_health`) are read off the template via `Minion.raw_attack` and `Minion.max_health`. Aura bonuses are computed on demand at every attack read, so they vanish the moment the source minion dies.

After `simulate_battle` returns, `MiniBGGame` only applies `(dmg_to_p0, dmg_to_p1)` to player health. Player boards are never mutated by battle.

## Board ordering invariants

- `BUY_SLOT_i`: `shop[i] → first empty hand slot`. Never touches `board`. Illegal when hand is full.
- `PLACE_HAND_i`: `hand[i] → board.append`. Illegal when board is full or `hand[i]` is empty.
- `SELL_BOARD_i`: list element removed at the chosen position; remaining minions shift left. Refunds 1 gold.
- `SELECT_ORDER_j` (env): `reorder_board(perm = PERMUTATIONS_4[j])` followed by submit. `reorder_board` keeps only positions `< k` (current board size) in the order specified by `perm`, and drops the empty tail. Only `k!` canonical perms are exposed via the legal mask (one per equivalence class).
- Battle summons: appended to the rightmost end of the battle side, only if `alive_count < 4`. Battle summons do not touch the persistent board.

## Action space (game core)

13 discrete actions (`Action` IntEnum). `legal_actions(state)` is phase-aware:

**Shop phase**:
- `BUY_SLOT_i` requires non-empty shop slot, hand not full, gold ≥ 3, `shop_actions_used < MAX_SHOP_ACTIONS`.
- `SELL_BOARD_i` requires position < `len(board)`, `shop_actions_used < MAX_SHOP_ACTIONS`.
- `PLACE_HAND_i` requires `hand[i]` non-empty, `len(board) < BOARD_SIZE`, `shop_actions_used < MAX_SHOP_ACTIONS`.
- `ROLL` requires gold ≥ 1, `shop_actions_used < MAX_SHOP_ACTIONS`.
- `LEVEL_UP` requires tier < 3, gold ≥ cost(tier), `shop_actions_used < MAX_SHOP_ACTIONS`.
- `FINISH` is always legal in shop phase. Flips `phase` to `ORDER`. Doesn't increment `shop_actions_used` and doesn't pass the turn.

**Order phase**: only `FINISH` is legal. Submits the order: sets `phase = DONE`, fires `ON_TURN_END`, passes turn (or triggers battle).

After `MAX_SHOP_ACTIONS` BUY/SELL/PLACE/ROLL/LEVEL_UP actions, the game auto-flips the player to the order phase; `FINISH` is then the only legal action.

`MiniBGGame.reorder_board(state, idx, perm)` is a free primitive (no action cost) used by the env to fuse reorder + submit. With **compact-after-permute** semantics, it accepts any of the 24 permutations of `(0,1,2,3)`: positions `< k` are taken in `perm` order and concatenated, positions `≥ k` are dropped.

## Action space (RL env)

37 discrete actions (`action_map.py`):

| Index range | Meaning                                                |
|-------------|--------------------------------------------------------|
| 0           | `ROLL`                                                 |
| 1           | `LEVEL_UP`                                             |
| 2..4        | `BUY_SLOT_0/1/2`                                       |
| 5..8        | `SELL_BOARD_0/1/2/3`                                   |
| 9..11       | `PLACE_HAND_0/1/2`                                     |
| 12          | `FINISH` (shop → order)                                |
| 13..36      | `SELECT_ORDER_0..23`: apply permutation `PERMUTATIONS_4[j]` and submit the order (in order phase only). |

`PERMUTATIONS_4` is the 24-element list of permutations of `(0,1,2,3)` in lexicographic order (`itertools.permutations`); index 0 is the identity. Only the **`k!` canonical perms** are legal in the order phase, where `k = len(board)`: a perm is canonical iff `perm[j] == j` for `j >= k`. This gives exactly one representative per equivalence class under compact-after-permute, so each legal SELECT_ORDER produces a distinct board layout. Exposing all 24 (with up to 24/k! redundant copies) was empirically harmful — DQN spread its gradient across redundant outputs and the argmax over reorder degenerated to a coin flip in early rounds.

`MiniBGEnv.legal_actions_mask` is phase-aware:
- **Shop phase**: projects `MiniBGGame.legal_actions(state)` onto slots 0..12 (`SELECT_ORDER_*` are forbidden).
- **Order phase**: only canonical SELECT_ORDER slots are legal — `1` for `k <= 1`, `2` for `k = 2`, `6` for `k = 3`, `24` for `k = 4`.

## Observation (RL env)

Fixed-size float32 vector, **self-centric** (the current player is always "me"):

- 10 globals: `round/15`, `my_hp/15`, `enemy_hp/15`, `gold/8`, `gold_cap/8`, `my_tier/3`, `enemy_tier/3`, `actions_left/10`, `my_board_count/4`, `has_initiative_if_equal_board_size`.
- 4 × 25 own board slots, 3 × 25 shop slots, 3 × 25 hand slots, 4 × 25 last-seen enemy board slots. Each 25-D slot vector encodes: presence, card_id one-hot (10 cards), tier one-hot (3), 4 stat scalars (base/bonus attack/health), `Taunt`, `Shield`, runtime `has_shield`, and one flag per ability trigger (`ON_BUY`, `ON_DEATH`, `AURA`, `ON_TURN_END`).
- 1 last-battle scalar = `(damage_dealt − damage_taken) / 7` from the previous round, from this player's perspective.
- 1 phase indicator: `0.0` for `SHOP`, `1.0` for `ORDER`. (`DONE` is transient and never observed by the acting player.)

Total: `OBS_DIM = 10 + 4·25 + 3·25 + 3·25 + 4·25 + 1 + 1 = 362`.

The enemy's board is **only** updated post-battle (last-seen snapshot). The enemy's hp and tier are read live from current state — those are public. The enemy's hand is never observed (hand is private information).

## Reward (RL env)

Terminal-only: +1 win, −1 loss, 0 draw, all from the perspective of the player whose action just produced the terminal state. Illegal actions return reward `INVALID_ACTION_REWARD = -1.0` and do not mutate state. Battle damage shaping is **not** applied to the env reward; `info["battle_signed"]` is exposed instead, and `AgentPerspectiveEnv.shaping_fn` (built by `make_minibg_shaping_fn`) attributes shaping symmetrically to whichever side the agent plays.
