# Patch packages (`data/bgcore/`)

Each Hearthstone Battlegrounds client build used for training/simulation is described by a **patch package**: a self-contained directory with catalog stats, hand-authored meta, and Python effect bindings. The shared rules engine (`bg_core`, `bg_recruitment`, `bg_combat`) stays patch-agnostic; only data and bindings change between builds.

Checkpoints are **not** compatible across patches (`card_emb` size = `len(templates)`, per-patch card index).

## Layout

```
data/bgcore/{version}_{build}/
  catalog.json    # auto-generated (HSJSON + HearthSim CardDefs)
  meta.json       # hand-edited (rotation tribes, pool copies, …)
  bindings.py     # hand-edited (card_id → Ability[])
```

Example: `data/bgcore/15_6_2_36393/` — patch `15.6.2`, build `36393`.

| File | Source | Contents |
|------|--------|----------|
| `catalog.json` | `scripts/build_minibg_patch_catalog.py` | All BG minions: tier, stats, race, mechanics, `isBaconPoolMinion`, golden upgrade linkage |
| `meta.json` | hand | `rotation_tribes`, `rotation_excluded_count`, `pool_copies_by_tier` |
| `bindings.py` | hand | `EFFECTS`, `TOKEN_IDS`, `GOLDEN_REWARD_IDS` — imports effect dataclasses from `src/bg_core/effects.py` |

Runtime entry point: `PatchContext.load(patch_dir)` in `src/bg_catalog/patch_context.py`.

Produces:

- `templates` / `descriptions` (always includes display `name`)
- `pool_ids` — non-golden tavern pool minions
- `effects`, `token_ids`, `golden_reward_ids`
- `card_id_to_dense`, `num_pool_indices` — for obs + network sizing
- `meta` — rotation and shared-pool copy counts

## Adding a new patch

### 1. Create directory

```bash
mkdir -p data/bgcore/19_6_0_74257
```

Naming: `{patch_with_underscores}_{build}` (e.g. `19.6.0` + `74257` → `19_6_0_74257`).

### 2. Generate `catalog.json`

Requires HearthSim `CardDefs.xml` for the target build and HearthstoneJSON `cards.json`:

```bash
python scripts/build_minibg_patch_catalog.py \
  --card-defs ~/hsdata/CardDefs.xml \
  --build 74257 --patch 19.6.0 \
  --out data/bgcore/19_6_0_74257/catalog.json
```

Offline (saved HSJSON):

```bash
python scripts/build_minibg_patch_catalog.py \
  --card-defs ~/hsdata/CardDefs.xml \
  --hsjson data/minibg/cards_74257_raw.json \
  --build 74257 --patch 19.6.0 \
  --out data/bgcore/19_6_0_74257/catalog.json
```

If `--out` is omitted, the script writes to `data/bgcore/{patch}_{build}/catalog.json` derived from `--patch` and `--build`.

### 3. Add `meta.json`

Copy from an existing patch and adjust rotation / pool sizes:

```json
{
  "rotation_tribes": ["BEAST", "DEMON", "DRAGON", "ELEMENTAL", "MECHANICAL", "MURLOC", "PIRATE"],
  "rotation_excluded_count": 2,
  "pool_copies_by_tier": { "1": 16, "2": 15, "3": 13, "4": 11, "5": 9, "6": 7 }
}
```

### 4. Write `bindings.py`

Start from a nearby patch (e.g. copy `15_6_2_36393/bindings.py`) and:

1. Keep shared tokens / golden reward ids that still exist in the new catalog.
2. Add or update `EFFECTS["CARD_ID"]` tuples for each pool minion that needs scripted behavior.
3. Use existing effect dataclasses in `src/bg_core/effects.py` — add new primitives to the engine only when a card truly needs one.

Keys are Hearthstone `card_id` strings (e.g. `BGS_004`), not slug aliases.

### 5. Validate coverage

```bash
python scripts/check_patch_coverage.py data/bgcore/19_6_0_74257
```

Reports:

- **errors** — pool/catalog mismatch, unknown `EFFECTS` keys, bad token/golden ids
- **warnings** — `BATTLECRY` / `DEATHRATTLE` in catalog without matching trigger in bindings (golden triple abilities are derived from normal-card bindings via `triple_effects`, not separate `TB_BaconUps_*` rows)
- **info** — pool cards with hook text (`Whenever`, `Each turn`, …) but no binding yet

Use `--fail-on-warning` in CI when bindings should be complete.

### 6. Wire training / eval

In YAML config:

```yaml
game:
  params:
    patch_dir: data/bgcore/19_6_0_74257
```

Training resolves `num_pool_indices` and `patch_build` from this path; checkpoints store `patch_build` and reject reload on mismatch.

### 7. Tests

```bash
pytest tests/test_patch_context.py tests/test_patch_coverage_script.py -q
python scripts/check_patch_coverage.py data/bgcore/19_6_0_74257
```

## Principles

- **Patch = data + bindings**, not engine forks.
- **Per-patch vocab** — no union card index across patches.
- **`PatchCardDescription.name`** is always set for UI/replay/debug.
- Reuse effect primitives; extend `bg_core` only for genuinely new mechanics (e.g. Reborn, Start of Combat for 74257).

## Related

- `src/bg_catalog/patch_context.py` — loader
- `src/training/patch_config.py` — `patch_dir` → agent `num_pool_indices` / `patch_build`
- `scripts/build_minibg_patch_catalog.py` — catalog builder
- `scripts/check_patch_coverage.py` — bindings coverage checker
