# Patch 74257 — retail fixes (normal minions)

Backlog тикетов на **соответствие catalog text ↔ engine** для **обычных** (non-golden) карт патча **19.6.0 / build 74257**.

**Out of scope:** golden / triple (`TB_BaconUps_*`), `implicit_triple_golden_effect` — отдельный backlog.  
**Связанный документ:** [patch_74257_tickets.md](./patch_74257_tickets.md) (infra / migration, помечена ✅).

**Источник текста карт:** `data/bgcore/19_6_0_74257/catalog.json`  
**Bindings:** `data/bgcore/19_6_0_74257/bindings.py`  
**Проверка coverage:** `python3 scripts/check_patch_coverage.py data/bgcore/19_6_0_74257`

### Сводка (2026-05-23)

| | Кол-во |
|---|--------|
| ✅ normal retail fixes | 7 / 7 |
| 🔶 golden / triple | out of scope |

**Легенда:** ✅ ok · 🔶 частично · ❌ не соответствует retail · P0/P1/P2 — приоритет

---

## Оглавление

| ID | Карта | Приоритет | Статус |
|----|-------|-----------|--------|
| [T-RN-01](#t-rn-01-qiraji-harbinger-bgs_112) | Qiraji Harbinger `BGS_112` | P0 | ✅ |
| [T-RN-02](#t-rn-02-imprisoner-bgs_014) | Imprisoner `BGS_014` | P0 | ✅ |
| [T-RN-03](#t-rn-03-herald-of-flame-bgs_032) | Herald of Flame `BGS_032` | P1 | ✅ |
| [T-RN-04](#t-rn-04-deflect-o-bot-bgs_071) | Deflect-o-Bot `BGS_071` | P1 | ✅ |
| [T-RN-05](#t-rn-05-gentle-djinni-bgs_121) | Gentle Djinni `BGS_121` | P1 | ✅ |
| [T-RN-06](#t-rn-06-scallywag-bgs_061) | Scallywag `BGS_061` | P2 | ✅ |
| [T-RN-07](#t-rn-07-legacy-pool-bindings) | Legacy pool (4 ids) | P2 | ✅ |

---

## Как читать тикет

| Поле | Значение |
|------|----------|
| **Catalog** | Текст из `catalog.json` (normal row). |
| **Сейчас** | Фактическое поведение в коде. |
| **Сделать** | Engine + binding + тест. |
| **Done when** | Интеграционный тест в `tests/test_patch_74257.py`. |

---

## T-RN-01: Qiraji Harbinger (`BGS_112`)

| | |
|---|---|
| **Статус** | ✅ — `ON_FRIENDLY_MINION_DIED` + `filter_victim_keyword=TAUNT` → `BuffDeadMinionNeighborsEffect` |
| **Приоритет** | P0 |
| **Tier** | 4 |

**Catalog:** After a friendly minion with **Taunt** dies, give its **neighbors** +2/+2.

**Сейчас:** `ON_PLACE` + `BuffRandomUniqueTribeFriendlies(count=3, +2/+2)` — battlecry Menagerie, не Qiraji.

```538:542:data/bgcore/19_6_0_74257/bindings.py
    "BGS_112": (
        Ability(
            Trigger.ON_PLACE,
            BuffRandomUniqueTribeFriendlies(count=3, attack=2, health=2),
        ),
    ),
```

**Сделать**

1. **Engine:** listener на смерть союзника с Taunt → buff соседей (index ±1 на board).
   - Trigger: `ON_FRIENDLY_MINION_DIED` (combat) + фильтр `Keyword.TAUNT` на умершем.
   - Effect (новый): e.g. `BuffNeighborsOfDeadMinionEffect(attack=2, health=2)` или reuse `AdjacentStatAura`-style one-shot в `battle.py` `_fire_friendly_minion_died_listeners`.
2. **Binding:** заменить `BGS_112` row; **не** трогать `BGS_082` / `BGS_083` (Menagerie Mug/Jug).
3. **Тест:** board `[taunt, harbinger, filler]` → taunt dies in combat → neighbors +2/+2; non-taunt death → no buff.

**Depends:** combat `ON_FRIENDLY_MINION_DIED` dispatch (уже есть для Soul Juggler).

---

## T-RN-02: Imprisoner (`BGS_014`)

| | |
|---|---|
| **Статус** | ✅ — `SummonEffect(token_id="BRM_006t")` |
| **Приоритет** | P0 |
| **Tier** | 2 |

**Catalog:** Deathrattle: Summon a **1/1 Imp**.

**Сейчас:** `SummonEffect(token_id="CS2_065")` — **Voidwalker** 1/3 Taunt, не Imp.

**Сделать**

1. Token: **`BRM_006t`** (1/1 Imp в catalog); добавить в `TOKEN_IDS` если ещё нет.
2. Binding: `"BGS_014": (Ability(Trigger.ON_DEATH, SummonEffect(token_id="BRM_006t", count=1)),)`.
3. **Тест:** Imprisoner dies → summoned minion 1/1 Demon Imp, не Taunt 1/3.

**Depends:** —

---

## T-RN-03: Herald of Flame (`BGS_032`)

| | |
|---|---|
| **Статус** | ✅ — `DealDamageLeftmostEnemyMinion(amount=3)` on overkill |
| **Приоритет** | P1 |
| **Tier** | 4 |

**Catalog:** **Overkill:** Deal 3 damage to the **left-most** enemy minion.

**Сейчас:** `ON_OVERKILL` + `DealDamageRandomEnemyMinion(amount=3)` — random enemy, не left-most.

**Сделать**

1. **Engine:** `DealDamageLeftmostEnemyMinion(amount=3)` или параметр `target=LEFTMOST` на существующий effect; dispatch в `_handle_overkill`.
2. **Binding:** заменить effect у `BGS_032`.
3. **Тест:** enemy board `[weak, strong]` → overkill hit damages **left** minion (index 0), не random.

**Depends:** —

---

## T-RN-04: Deflect-o-Bot (`BGS_071`)

| | |
|---|---|
| **Статус** | ✅ — `Ability.combat_only=True`; shop dispatch skip |
| **Приоритет** | P1 |
| **Tier** | 3 |

**Catalog:** Whenever you summon a Mech **during combat**, gain +1 Attack and **Divine Shield**.

**Сейчас:** `ON_FRIENDLY_MINION_SUMMONED` — срабатывает и в **shop** (`fire_shop_friendly_summoned` в `shop_triggers.py`), что противоречит тексту.

**Сделать**

1. **Engine (выбрать один):**
   - **A:** не вызывать `ON_FRIENDLY_MINION_SUMMONED` listeners из shop path для карт с combat-only semantics; **или**
   - **B:** новый trigger `ON_COMBAT_FRIENDLY_MINION_SUMMONED` + combat dispatch only; **или**
   - **C:** флаг на `Ability` / effect: `combat_only=True`, shop dispatch skip.
2. **Binding:** оставить Mech filter + shield/attack; привязать к combat-only path.
3. **Тест:** shop: play Mech → Deflect-o-Bot **без** buff/shield; combat summon Mech → +1 Attack + Shield.

**Depends:** —

> **Решение (рекомендация):** **C** — `Ability.combat_only: bool` — минимальный diff, не ломает Pack Leader / Mama Bear в shop.

---

## T-RN-05: Gentle Djinni (`BGS_121`)

| | |
|---|---|
| **Статус** | ✅ — `SummonRandomAndCopyToHandEffect` → board + `combat_hand_adds` |
| **Приоритет** | P1 |
| **Tier** | 6 |

**Catalog:** Deathrattle: Summon another random **Elemental** and **add a copy of it to your hand**.

**Сейчас:** `SummonRandomMinionEffect(count=1, race_filter=Race.ELEMENTAL)` — board only.

**Сделать**

1. **Engine:** combined effect e.g. `SummonRandomAndCopyToHandEffect(race_filter=Race.ELEMENTAL)` **или** два последовательных шага в deathrattle dispatch (summon → copy same `card_id` to hand if slot free).
2. **Binding:** заменить `BGS_121` row.
3. **Тест:** Djinni dies in combat → 1 Elemental on board + same id in hand (if hand slot); hand full → summon only.

**Depends:** combat hand add path (аналог Nat Pagle `combat_hand_adds` / shop hand).

---

## T-RN-06: Scallywag (`BGS_061`)

| | |
|---|---|
| **Статус** | ✅ — `SummonEffect(..., attack_immediately=True)` |
| **Приоритет** | P2 |
| **Tier** | 1 |

**Catalog:** Deathrattle: Summon a 1/1 Pirate. **It attacks immediately.**

**Сейчас:** `SummonEffect(token_id="BGS_061t", count=1)` — token появляется, но не бьёт сразу.

**Сделать**

1. **Engine:** `SummonEffect(..., attack_immediately=True)` или post-summon hook в combat DR queue (как Yo-Ho-Ogre follow-up attack).
2. **Binding:** флаг на `BGS_061` summon.
3. **Тест:** Scallywag dies → token summons → token выполняет одну атаку в том же combat до следующего swing phase.

**Depends:** combat summon + attack scheduling в `battle.py`.

---

## T-RN-07: Legacy pool bindings

| | |
|---|---|
| **Статус** | ✅ — 4 legacy bindings; `check_patch_coverage` 0 warnings |
| **Приоритет** | P2 |

**Карты** (normal, `isBaconPoolMinion`, не `BGS_*`):

| ID | Name | Catalog text (кратко) |
|----|------|------------------------|
| `FP1_024` | Unstable Ghoul | Deathrattle: deal 1 damage to all minions |
| `YOD_026` | Fiendish Servant | Deathrattle: give this minion's Attack to a random friendly minion |
| `BT_010` | Felfin Navigator | Battlecry: give your Murlocs +1/+1 |
| `DMF_533` | Ring Matron | Deathrattle: summon 3/6 minion |

**Сделать**

1. Bindings в `bindings.py` (reuse existing effects где возможно):
   - Ghoul → AOE damage on `ON_DEATH` (новый или `DealDamageAllMinionsEffect`).
   - Fiendish Servant → `TransferAttackToRandomFriendlyEffect` on `ON_DEATH`.
   - Felfin Navigator → `BuffAllFriendlyOfTribe(MURLOC, +1/+1)` on `ON_PLACE`.
   - Ring Matron → `SummonEffect(token_id="DMF_533t", count=2)` (two 3/2 Imps).
2. **Тесты:** по одному smoke на карту или parametrized legacy test.
3. `check_patch_coverage.py` → **0 warnings** (optional `--fail-on-warning` в CI).

**Depends:** —

---

## Карты, проверенные как OK (normal)

Не требуют тикета в этом backlog (spot-check + audit 2026-05-23):

- **Keyword-only:** `BGS_034` Bronze Warden (Reborn), `BGS_039` Taunt, `BGS_049` sell 3g (catalog parse), `BGS_106` Reborn+Taunt, `BGS_119` Windfury+Shield, `BGS_131` Poisonous.
- **Shop/combat hooks с тестами:** Nomi, Kalecgos (battlecry gate), Hangry Dragon, Amalgadon, Murozond (plain copy), Faceless (plain transform), Sellemental, Stasis, Deck Swabbie, Hoggarr, Soul Devourer (normal consume), Wildfire (`ON_OVERKILL` + excess adjacent — совпадает с «attacks and kills» при excess damage).
- **Остальные ~60 `BGS_*`:** bindings и dispatch согласованы с catalog на уровне trigger/effect class (без golden).

При регрессии — добавить card-specific test, не новый infra-тикет.

---

## Порядок работ

```
RN-01 (Qiraji) → RN-02 (Imprisoner token) → RN-03..05 (combat fidelity) → RN-06 → RN-07 (legacy)
```

**Критический path до «lobby без грубых ошибок»:** T-RN-01, T-RN-02.

---

## Ссылки

- [patch_74257_tickets.md](./patch_74257_tickets.md) — migration / engine infra  
- [patch_package.md](./patch_package.md) — формат catalog / bindings  
- Effects: `src/bg_core/effects.py`  
- Combat: `src/bg_combat/battle.py`  
- Shop triggers: `src/bg_recruitment/shop_triggers.py`
