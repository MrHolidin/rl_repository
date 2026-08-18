# 36.2.0 / build 248348 — package status

Built from HearthstoneJSON alone (`--hsjson`, no `CardDefs.xml`): modern JSON
carries `techLevel`, `isBattlegroundsPoolMinion` and `battlegroundsPremiumDbfId`
itself. The 2021 dumps do not — `techLevel` is null even on real tavern minions
there — so the older packages keep being built with the XML.

## What the catalog holds

| Section | Rows | Notes |
|---|---|---|
| `minions` | 2492 | everything ever printed with a tavern tier; 1166 golden printings |
| — of them in the live pool | 274 | `isBaconPoolMinion`, tiers 1–7 |
| `tavernSpells` | 75 | `BATTLEGROUND_SPELL`, school TAVERN, tiers 1–7 |
| `trinkets` | 390 | card *type* `BATTLEGROUND_TRINKET` — no `isBattlegrounds…` flag exists |
| `heroes` | 121 | |
| `darkGifts` | 43 | season 14 mechanic |

Tribes on live pool cards, from the data: Beast, Demon, Dragon, Elemental,
Mech, Murloc, Naga, Pirate, Quilboar, Undead, plus 6 Amalgams (`ALL`) and 31
tribeless.

## Numbers in `meta.json` that are NOT verified

Everything above comes from the card data. These do not, and no source has been
checked for them yet — they are placeholders carried over from 19.6.0 shape:

- `rotation_excluded_count` (how many of the ten tribes sit out a lobby);
- `pool_copies_by_tier`, including the tier-7 entry, which is invented;
- `level_up_costs` for tiers 5 and 6 (the step to 6 and to 7);
- everything the `ruleset` block leaves at its default: gold curve, gold cap,
  buy/sell/roll costs, starting health (modern heroes start at 30 plus armor,
  not the classic 40), damage cap.

Confirm them before anything is trained on this package. The engine will happily
simulate wrong numbers.

## Not in the package yet

`bindings.py` is empty of effects, so every card here is a vanilla body. That is
the honest starting state: `python scripts/check_patch_coverage.py
data/bgcore/36_2_0_248348` prints the work queue.
