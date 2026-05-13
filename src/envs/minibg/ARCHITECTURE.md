# Mini BG 1v1 — module architecture

Two layers:
- **Game core** (`actions.py`, `state.py`, `effects.py`, `cards.py`, `patch_catalog.py`, `battle.py`, `game.py`) — pure rules.
- **RL wrapper** (`action_map.py`, `obs.py`, `env.py`) — fixed-size action space, self-centric vector observation, terminal-only reward.

## Files

- `effects.py` — `Keyword`, `Trigger`, frozen-dataclass effects (`StatAura`, `TribalOtherStatAura`, `KeywordStatAura`, `AdjacentStatAura`, `BuffAdjacentBattlecry`, `KangorSummonCopy`, `BattlecryMultiplierAura` / `DeathrattleMultiplierAura` / `SummonMultiplierAura`, `HeroImmuneAura`, …), and `Ability(...)`.
- `state.py` — `Minion`, `PlayerState` (`health`, `gold`, `tavern_tier`, **`next_tier_up_cost`**, `board`, `shop`, `hand`, `phase`, `shop_actions_used`, `hero_damage_taken_total`, pending-choice fields), `MiniBGState`. `PlayerPhase` is `SHOP | ORDER | DONE`.
- `cards.py` — `CARD_TEMPLATES` declarative pool, `make_minion(card_id)`, `shop_pool_for_tier(tier)`.
- `patch_catalog.py` — loads `data/minibg/bg_patch_15_6_2_36393_catalog.json`: 186 Hearthstone BG minions with **tavern tier** from HearthSim `CardDefs.xml` (commit `e6cdbc5`, build 36393) merged with HearthstoneJSON card text/stats. Regenerate via `scripts/build_minibg_patch_catalog.py`.
- `actions.py` — `Action` IntEnum (28 discrete game actions: BUY×3, SELL×4, ROLL, LEVEL_UP, FINISH, PLACE_HAND×3, MAGNET_HAND×BOARD×12, DISCOVER_PICK×3) and gameplay constants (`SHOP_SIZE=3`, `BOARD_SIZE=4` shop/placement cap, `HAND_SIZE=3`). Combat summoning uses `battle.COMBAT_BOARD_MAX=7` (BG board space).
- `battle.py` — event-driven combat (`BeginAttackExchange`, `DamageStrike` → `ShieldLost` / `DamageDealt` / `Overkill`, `AttackCompleted`, `MinionDied`, `MinionSummoned`). Strikes use `attack_with_auras`; in death resolution `attack_value(..., death_resolution=True)` and `health_aura_bonus(..., death_resolution=True)` **freeze** continuous stat auras (no tribal/keyword/adjacent/global contributions during `ON_DEATH` / Kangor windows). **Health auras** (`StatAura.health`, `TribalOtherStatAura`, `KeywordStatAura`, `AdjacentStatAura`) use `health_aura_snapshot` and resync after damage/summons so current/max drop when a source dies (e.g. Mal'Ganis). **Adjacency** (`AdjacentStatAura`): only board index ±1 from the source; **corpses keep slots** (no nearest-living retarget). **Baron** / **Khadgar** / Kangor / board-cap behavior unchanged. **Initiative**, **Windfury**, **Zapp**, **Cleave**, **Charge** as before (`simulate_battle(..., death_log=, mech_death_log=, p0_survivors_out=)`).
- `game.py` — `MiniBGGame(TurnBasedGame[MiniBGState])`: shop loop, hand mechanic, two-phase shop turn (`shop` → `order`), `ON_BUY` / `ON_PLACE` / `AFTER_FRIENDLY_MINION_PLACED` / `ON_TURN_END` dispatch, **Magnetic** (`MAGNET_HAND_h_BOARD_b`): merge hand Magnetic Mech onto an existing board Mech (stats add, keywords combine incl. Divine Shield, target keeps board buffs, `ON_DEATH` from magnet queued after existing DRs; no `ON_PLACE` / battlecry from the magnetic piece), hero damage + `hero_damage_taken_total`, Mal'Ganis-style hero immune (`HeroImmuneAura`), **Brann** (`BattlecryMultiplierAura`): each `ON_PLACE` effect iterates with **fresh** product `Π` over `player.board` each repetition (normal `factor=2`, golden `3`); **not** applied to `AFTER_FRIENDLY_MINION_PLACED`, round orchestration, battle invocation, `reorder_board` primitive (compact-after-permute).
- `action_map.py` — 52-action env layout: `ROLL` / `LEVEL_UP` / `BUY_SLOT_*` / `SELL_BOARD_*` / `PLACE_HAND_*` / `MAGNET_HAND_*_BOARD_*` / `DISCOVER_PICK_0..2` / `FINISH` / `SELECT_ORDER_0..23`. Holds the precomputed permutation table and the env→game action mapper.
- `obs.py` — fixed-size vector observation (10 globals + 4·SLOT_DIM own board + 3·SLOT_DIM shop + 3·SLOT_DIM hand + 4·SLOT_DIM last-seen enemy board + 1 last-battle scalar + 1 phase indicator + 10-dim pending-choice vector for Discover/Adapt modals).
- `env.py` — `MiniBGEnv(TurnBasedEnv)`: applies actions, on `SELECT_ORDER_*` calls `reorder_board(perm)` then submits the order via `apply_action(FINISH)` from the order phase. Tracks last-seen enemy board and signed last-battle damage delta per player, emits self-centric obs. On battle resolution emits `info["battle_signed"] = (signed_p0, signed_p1)` for symmetric per-player shaping; the env's own `step().reward` carries only terminal win/loss/draw (shaping is computed in `src.training.agent_perspective_env.AgentPerspectiveEnv`). Read-only `state` / `game` for scripted opponents. Optional `replay_path` / `replay_meta`: JSONL per-step snapshots via `replay.py` (replay format `2` includes `hand` and `phase`). Training: the trainer consumes `AgentPerspectiveEnv` which wraps `MiniBGEnv`, drives opponent moves until the agent must act again, and attributes terminal + shaping reward in the agent's zero-sum perspective.
- `heuristic_bots/` — scripted opponents (`RandomBot`, `TempoBot`, …) + `tournament.run_tournament`. CLI: `python -m src.envs.minibg.heuristic_bots` or `python scripts/minibg_tournament.py`. The shared `_finish(env)` helper is phase-aware: in shop phase it auto-places pending hand cards then issues `FINISH`; in order phase it picks a permutation via `choose_final_order`.
- **Shared eval / matches** — `MiniBGEnv` follows the same `TurnBasedEnv` contract as Connect4/Othello (`legal_actions_mask`, `current_player_token`, `winner` in `±1` / `0`), so `src.utils.match.play_single_game` / `play_match` work with multi-step shop turns (each loop picks the actor whose `current_player_token` matches `agent_token`). Use `play_match(..., game_id="minibg")` when no custom `env` is passed, or `make_game("minibg", ...)`. Heuristic bots implement `choose_action(env)`; for `play_match` use `MiniBGHeuristicAgent` from `heuristic_bots.agent_adapter` (set via `set_env`). `src.evaluation.eval_checkpoints.build_opponents_from_names(..., game_id="minibg")` accepts keys from `default_bot_constructors()` (e.g. `tempo`, `buffer_t2`, `random`).

## Effect pattern

Static keywords (`Taunt`, `Shield`) live on `Minion.keywords: frozenset[Keyword]`. The battle code reads them directly during target selection and damage application — these keywords are not "abilities".

Triggered and continuous effects live on `Minion.abilities: tuple[Ability, ...]`. Each `Ability` carries:
- a `Trigger` (…, `ON_FRIENDLY_MECH_DIED` is **combat-only** / ignored in shop),
- a typed `Effect` dataclass,
- optional `filter_race` (for `AFTER_FRIENDLY_MINION_PLACED`).

Dispatch happens in these places:

| Trigger        | Where                                | Resolver                                                    |
|----------------|--------------------------------------|-------------------------------------------------------------|
| `ON_BUY`       | `MiniBGGame._fire_on_buy`            | match on `effect` type; `BuffRandomFriendly` picks a random target from `board` only. The bought minion sits in hand at trigger time. |
| `ON_PLACE`     | `MiniBGGame._fire_on_place`        | runs on the minion just appended to the board (play / battlecry timing). |
| `AFTER_FRIENDLY_MINION_PLACED` | `MiniBGGame._fire_after_friendly_minion_placed` | left-to-right over full board **after** `ON_PLACE` on the placed card; respects `filter_race`. |
| `ON_DEATH`     | `battle._fire_deathrattle`           | `SummonEffect` appends a token (no stat auras applied during this window beyond `attack_value` rules). |
| `ON_FRIENDLY_MECH_DIED` | `battle._fire_kangor_listeners` | Kangor-style: shallow-copy the dying Mech template (`copy.copy`) onto the board if there is room. |
| `ON_TURN_END`  | `MiniBGGame._fire_on_turn_end`       | left-to-right `BuffRandomFriendly` at submit-order time. |
| `ON_TURN_START`| `MiniBGGame._fire_on_turn_start`     | after `round_number` increments, **before** shop reroll: board L→R, then hand; `BuffSelf` / `BuffRandomFriendly`. Continuous **auras** are not triggers; they are reflected whenever stats are read during shop (and do not execute extra code at this boundary). |
| `AURA`         | `battle.attack_value` / `health_aura_bonus` with `death_resolution=False` (strike phase) | typed continuous buffs from other **alive** friendlies: `StatAura` (global other), `TribalOtherStatAura` (your other tribe), `KeywordStatAura` (keyword gate), `AdjacentStatAura` (fixed slot ±1). Same helpers return **no aura contribution** when `death_resolution=True`.          |

Adding a new card = one entry in `CARD_TEMPLATES` (data only).
Adding a new effect *type* = one frozen dataclass in `effects.py` + one branch in the matching dispatcher.

## Randomness

`MiniBGGame` owns a single `np.random.Generator`. `apply_action` is **not pure**: it consumes RNG for shop generation, Buffer's random target, battle target selection, and round-1 initiative.
This is by design for v0; if MCTS-style search is added later, RNG state can be pushed into `MiniBGState`.

## Round flow

Each round is split into a shop phase and an order phase per player.

**Shop phase** (`PlayerPhase.SHOP`): the active player issues any sequence of `BUY` / `SELL` / `PLACE` / `MAGNET` / `ROLL` / `LEVEL_UP` actions; each costs one of `MAX_SHOP_ACTIONS` budget slots. None of these actions pass the turn. `FINISH` ends the shop phase by flipping `phase` → `ORDER` (also without passing the turn). If the action budget hits the cap, `apply_action` auto-flips to `ORDER`.

**Order phase** (`PlayerPhase.ORDER`): only the env-level `SELECT_ORDER_j` action is legal. The env layer applies `reorder_board(state, idx, perm)` and then calls `apply_action(FINISH)`; from order phase, `FINISH`:
1. Sets `phase = DONE`.
2. Fires `ON_TURN_END` triggers on the now-finalized board.
3. Either passes the turn to the other player (if they are not yet `DONE`) or resolves the battle.

**Battle resolution** (`_resolve_battle_and_advance`):
1. `simulate_battle` on deep copies of both boards (input boards unchanged).
2. Replace each player's persistent `board` with combat **survivors** (truncated to `BOARD_SIZE`).
3. Subtract damage from each player's health.
4. Terminal check (any player ≤ 0 hp, or round 20 finished).
5. Otherwise increment `round_number`, then **discount** each player's `next_tier_up_cost` by 1 (min 0) if below max tier, restore gold (`gold_for_round`), clear `shop_actions_used`, set both phases to `SHOP`, fire **`ON_TURN_START`** per player (board L→R, then hand), then reroll both shops. **Hand persists across rounds.**

## Battle copy semantics

`simulate_battle` builds `BattleMinion(template, …, instance_id)` from each `Minion`. Strike damage uses **alive** others for aura contributions. During deathrattle / Kangor resolution, `attack_value` / `health_aura_bonus` use `death_resolution=True` (frozen stats); summoned minions do not inherit live-board auras until the next normal strike phase.

After `simulate_battle`, the core game applies hero damage, then copies **alive** combat minions (in board scan order, truncated to `BOARD_SIZE`) onto each player's persistent `board` via `persist_shop_board_from_side` (Divine Shield re-arms on the copy). Tokens and Khadgar-duplicated summons persist if they survived combat. Input `Minion` lists are not mutated by `simulate_battle`.

**Shop-phase recruitment** after a non-terminal battle: round counter and gold are updated first; **`ON_TURN_START`** runs on each player's board (left-to-right) then hand, then `_refresh_shop` runs (free reroll).

## Board ordering invariants

- `BUY_SLOT_i`: `shop[i] → first empty hand slot`. Never touches `board`. Illegal when hand is full.
- `PLACE_HAND_i`: `hand[i] → board.append`. Illegal when board is full or `hand[i]` is empty.
- `MAGNET_HAND_h_BOARD_b`: requires `hand[h]` non-empty with `Keyword.MAGNETIC` and Mech tribe, `board[b]` a Mech (`MECHANICAL` or `ALL`), `shop_actions_used < MAX_SHOP_ACTIONS`. Merges into `board[b]` in place (hand slot cleared); does not append a minion or fire `ON_PLACE` on the magnetic card.
- `SELL_BOARD_i`: list element removed at the chosen position; remaining minions shift left. Refunds 1 gold.
- `SELECT_ORDER_j` (env): `reorder_board(perm = PERMUTATIONS_4[j])` followed by submit. `reorder_board` keeps only positions `< k` (current board size) in the order specified by `perm`, and drops the empty tail. Only `k!` canonical perms are exposed via the legal mask (one per equivalence class).
- Battle summons: appended to the rightmost end of the combat side while `alive_count < COMBAT_BOARD_MAX` (7). Surviving combat minions (including summons) are copied to the persistent board after battle, up to `BOARD_SIZE` (4), in combat scan order.

## Action space (game core)

25 discrete actions (`Action` IntEnum). `legal_actions(state)` is phase-aware:

**Shop phase**:
- `BUY_SLOT_i` requires non-empty shop slot, hand not full, gold ≥ 3, `shop_actions_used < MAX_SHOP_ACTIONS`.
- `SELL_BOARD_i` requires position < `len(board)`, `shop_actions_used < MAX_SHOP_ACTIONS`.
- `PLACE_HAND_i` requires `hand[i]` non-empty, `len(board) < BOARD_SIZE`, `shop_actions_used < MAX_SHOP_ACTIONS`.
- `MAGNET_HAND_h_BOARD_b` requires `hand[h]` Magnetic Mech, `b < len(board)`, `board[b]` a Mech, `shop_actions_used < MAX_SHOP_ACTIONS`.
- `ROLL` requires gold ≥ 1, `shop_actions_used < MAX_SHOP_ACTIONS`.
- `LEVEL_UP` requires tier `< MAX_TIER` (6), gold ≥ ``player.next_tier_up_cost``, `shop_actions_used < MAX_SHOP_ACTIONS`. After each battle, `next_tier_up_cost` decreases by 1 (min 0) until the player levels; on tier-up it resets to the **base** for the new step from `LEVEL_UP_COSTS` (wiki.gg table: 5,7,8,11,11).
- `FINISH` is always legal in shop phase. Flips `phase` to `ORDER`. Doesn't increment `shop_actions_used` and doesn't pass the turn.

**Order phase**: only `FINISH` is legal. Submits the order: sets `phase = DONE`, fires `ON_TURN_END`, passes turn (or triggers battle).

After `MAX_SHOP_ACTIONS` BUY/SELL/PLACE/MAGNET/ROLL/LEVEL_UP actions, the game auto-flips the player to the order phase; `FINISH` is then the only legal action.

`MiniBGGame.reorder_board(state, idx, perm)` is a free primitive (no action cost) used by the env to fuse reorder + submit. With **compact-after-permute** semantics, it accepts any of the 24 permutations of `(0,1,2,3)`: positions `< k` are taken in `perm` order and concatenated, positions `≥ k` are dropped.

## Action space (RL env)

49 discrete actions (`action_map.py`):

| Index range | Meaning                                                |
|-------------|--------------------------------------------------------|
| 0           | `ROLL`                                                 |
| 1           | `LEVEL_UP`                                             |
| 2..4        | `BUY_SLOT_0/1/2`                                       |
| 5..8        | `SELL_BOARD_0/1/2/3`                                   |
| 9..11       | `PLACE_HAND_0/1/2`                                     |
| 12..23      | `MAGNET_HAND_h_BOARD_b` (`h∈{0,1,2}`, `b∈{0,1,2,3}`) |
| 24          | `FINISH` (shop → order)                                |
| 25..48      | `SELECT_ORDER_0..23`: apply permutation `PERMUTATIONS_4[j]` and submit the order (in order phase only). |

`PERMUTATIONS_4` is the 24-element list of permutations of `(0,1,2,3)` in lexicographic order (`itertools.permutations`); index 0 is the identity. Only the **`k!` canonical perms** are legal in the order phase, where `k = len(board)`: a perm is canonical iff `perm[j] == j` for `j >= k`. This gives exactly one representative per equivalence class under compact-after-permute, so each legal SELECT_ORDER produces a distinct board layout. Exposing all 24 (with up to 24/k! redundant copies) was empirically harmful — DQN spread its gradient across redundant outputs and the argmax over reorder degenerated to a coin flip in early rounds.

`MiniBGEnv.legal_actions_mask` is phase-aware:
- **Shop phase**: projects `MiniBGGame.legal_actions(state)` onto slots 0..24 (`SELECT_ORDER_*` are forbidden).
- **Order phase**: only canonical SELECT_ORDER slots are legal — `1` for `k <= 1`, `2` for `k = 2`, `6` for `k = 3`, `24` for `k = 4`.

## Observation (RL env)

Fixed-size float32 vector, **self-centric** (the current player is always "me"):

- 10 globals: `round/20`, `my_hp/30`, `enemy_hp/30`, `gold/10`, `gold_cap/10`, `my_tier/MAX_TIER`, `enemy_tier/MAX_TIER`, `actions_left/20`, `my_board_count/4`, `has_initiative_if_equal_board_size`.
- 4 × 42 own board slots, 3 × 42 shop slots, 3 × 42 hand slots, 4 × 42 last-seen enemy board slots. Each SLOT_DIM slot vector encodes: presence, toy `card_id` one-hot (10), tavern **tier** one-hot (6 tiers), 4 stat scalars, **tribe** one-hot (none + 4 tribes + all-tribes), keywords (`Taunt`, `Shield`, `Windfury`, `Poisonous`, `Charge`, `Magnetic`), runtime `has_shield`, and ability-trigger flags (`ON_BUY`, `ON_DEATH`, `AURA`, `ON_TURN_END`, `ON_PLACE`, `AFTER_FRIENDLY_MINION_PLACED`, `ON_FRIENDLY_MECH_DIED`, `ON_TURN_START`).
- 1 last-battle scalar = `(damage_dealt − damage_taken) / 7` from the previous round, from this player's perspective.
- 1 phase indicator: `0.0` for `SHOP`, `1.0` for `ORDER`. (`DONE` is transient and never observed by the acting player.)

Total: `OBS_DIM = 10 + 14·SLOT_DIM + 2` with `SLOT_DIM = 41` (see `obs.py`).

The enemy's board is **only** updated post-battle (last-seen snapshot). The enemy's hp and tier are read live from current state — those are public. The enemy's hand is never observed (hand is private information).

## Reward (RL env)

Terminal-only: +1 win, −1 loss, 0 draw, all from the perspective of the player whose action just produced the terminal state. Illegal actions return reward `INVALID_ACTION_REWARD = -1.0` and do not mutate state. Battle damage shaping is **not** applied to the env reward; `info["battle_signed"]` is exposed instead, and `AgentPerspectiveEnv.shaping_fn` (built by `make_minibg_shaping_fn`) attributes shaping symmetrically to whichever side the agent plays.
