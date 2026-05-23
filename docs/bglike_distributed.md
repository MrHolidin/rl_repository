# BG distributed training stack

Сжатая документация по связке **distributed PPO + MiniBG/BGLike + bg_core + env + structured model + league**.

Для общего пайплайна (run_dir, callbacks, single-process) см. [pipeline.md](pipeline.md).  
Для правил и action space MiniBG см. [../src/envs/minibg/ARCHITECTURE.md](../src/envs/minibg/ARCHITECTURE.md).

---

## Назначение

Один distributed trainer обучает structured PPO на Battlegrounds-подобных играх:

| Режим | `game.id` | Игроки | Назначение |
|-------|-----------|--------|------------|
| **MiniBG** | `minibg` | 2 (1v1) | быстрый sandbox: rules, policy head, league loop |
| **BGLike** | `bglike` | 8 (lobby) | целевая постановка: placement reward, multi-seat control |

Оба режима используют общий rules stack (`bg_core`, `bg_recruitment`, …), один agent class (`MiniBGPPOStructuredAgent`) и один distributed loop (`DistributedTrainer`).

---

## Архитектура слоёв

```
bg_core              Minion, Race, Effect DSL (Keyword, Trigger, Ability, …)
bg_catalog           карточный пул (patch catalog)
bg_recruitment       shop, place, triples, discover, shop triggers
bg_player_turn       движок хода игрока (shop actions)
bg_lobby             8p lobby: pairing, combat round, shared pool, eliminations
bg_combat            симуляция боя
        ↓
game                 MiniBGGame (2p)  /  BGLikeGame (8p)
        ↓
RL wrapper           action_map, obs, env / lobby_env
        ↓
training perspective AgentPerspectiveEnv / BGLikeAgentPerspectiveEnv
        ↓
agent + model        MiniBGPPOStructuredAgent + structured actor-critic
        ↓
distributed          workers (CPU rollouts) + host (GPU PPO + league)
```

**Важно:** MiniBG и BGLike больше не дублируют rules engine целиком. Оба опираются на `bg_recruitment`, `bg_player_turn`, `bg_lobby.player`. MiniBG — упрощённый 2p loop с собственным `MiniBGState`; BGLike — полный 8p lobby с eliminations и placement.

### Ключевые модули

| Слой | MiniBG | BGLike |
|------|--------|--------|
| Game | `src/envs/minibg/game.py` | `src/envs/bglike/game.py` |
| Env | `MiniBGEnv` | `BGLobbyMultiCurrentEnv` (`lobby_env.py`) |
| Obs dim | 736 | 1359 |
| Terminal reward | ±1 win/loss/draw | placement in [-1, 1] |
| Opponents | 1 на игру | независимый sample на каждый non-current seat |
| League record | pairwise 2p | full-lobby placements |

---

## Запуск

### Distributed (основной путь)

```bash
python -m src.cli.train_distributed \
    --config configs/bglike/ppo_structured_dist.yaml \
    --run_dir runs/bglike/dist_ppo_026
```

Конфиг = обычный train YAML **+ блок `distributed:`**:

```yaml
distributed:
  workers: 8              # число CPU worker-процессов
  worker_device: cpu      # device для inference на workers
  # checkpoint: path/to/init.pt   # optional warm start
```

Host device задаётся в `agent.params.device` (обычно `cuda`).

### Примеры конфигов

| Файл | Что |
|------|-----|
| `configs/minibg/ppo_structured_dist.yaml` | 2p sandbox, `minibg_structured` |
| `configs/bglike/ppo_structured_dist.yaml` | 8p, v1 model, `num_current_seats: 1` |
| `configs/bglike/ppo_structured_v2.yaml` | 8p, v2 model, можно `num_current_seats: 4` |

Single-process (без workers): `python -m src.cli.train --config …` — см. [pipeline.md](pipeline.md).

---

## Distributed training

### Host / worker split

```
┌─────────────────────────────────────────────────────────┐
│  HOST (GPU)                                             │
│  - learner weights                                      │
│  - PPO update после каждого раунда                      │
│  - LeagueController (рейтинги, frozen pool)             │
│  - checkpoint / metrics / status callbacks              │
└───────────────┬─────────────────────────────────────────┘
                │  load / sync_league / play / stop
    ┌───────────┼───────────┬───────────┐
    ▼           ▼           ▼           ▼
 Worker 0    Worker 1    Worker 2   …
 (CPU)       (CPU)       (CPU)
 rollouts    rollouts    rollouts
```

**Worker protocol** (`src/training/distributed_trainer.py`):

| Host → Worker | Назначение |
|---------------|------------|
| `("load", sd_bytes)` | обновить веса learner |
| `("sync_league", LeagueSyncState)` | frozen pool + win rates / TrueSkill |
| `("play", round_idx)` | собрать игры до `rollout_steps` |
| `("stop",)` | завершить процесс |

| Worker → Host | Назначение |
|---------------|------------|
| `("rollout", n_games, n_steps, …, payload, outcomes)` | буфер + `List[GameRecord]` |

### Round loop

1. Host рассылает `sync_league` (актуальный frozen pool + рейтинги).
2. Host рассылает `load` (последние веса learner).
3. Host шлёт `play` всем workers параллельно.
4. Workers собирают rollouts structured policy (`act_structured` / `step_structured`).
5. Host мержит буферы, делает PPO update на GPU.
6. Host обновляет `LeagueController` из `GameRecord` outcomes.
7. При пересечении `step // checkpoint_interval` — сохранение в pool.
8. Callbacks: `status.json`, `metrics.csv`, `self_play_frozen.json`.

**Критично:** league stats обновляются **только на host**. Workers stateless по рейтингам — только применяют snapshot из `sync_league`. Локальное обновление статистики на worker = bug.

### Opponent sampling

На каждый opponent (или seat в BGLike):

1. `current_self_fraction` → текущие веса learner (`slot_id = -1`)
2. `past_self_fraction` → frozen checkpoint, PFSP-weighted (если pool не пуст)
3. Остаток → scripted bot

| Игра | Как сэмплируется |
|------|------------------|
| MiniBG | 1 opponent на 2p игру |
| BGLike | независимый sample на каждый seat, не controlled learner (`8 - num_current_seats`) |

Когда frozen pool пуст, `past_fraction` перетекает в scripted (см. `decide_opponent_kind`).

---

## Environment и perspective

### MiniBG (1v1)

`MiniBGEnv` → `AgentPerspectiveEnv`:

- opponent ходит автоматически до хода learner;
- terminal reward: +1 / −1 / 0;
- battle shaping: `info["battle_signed"]` × `battle_damage_shaping` через `make_minibg_shaping_fn`;
- env reward terminal-only; shaping идёт через perspective wrapper.

### BGLike (8p lobby)

`BGLobbyMultiCurrentEnv` → `BGLikeAgentPerspectiveEnv` (`src/training/bglike_perspective.py`):

- 8 seats, у каждого controller: learner / frozen copy / scripted bot;
- **`num_current_seats`** — сколько seats играет текущая политика (shared weights);
- **segment boundaries**: смена learner seat или elimination seat;
- terminal reward = `placement_reward(place)` в [-1, 1]: 1-е место → +1, 8-е → −1;
- battle shaping: `info["battle_signed_seat"]` × scale;
- league outcome пишется **один раз на lobby end**, не на каждый segment.

Placement формулы (`src/envs/bglike/placement.py`):

- reward: `(9 - 2*place) / 7`
- league score: `(reward + 1) / 2` → [0, 1]
- pairwise vs opponent: lower place wins (1.0 / 0.0 / 0.5)

### Structured control

Distributed path использует **structured actions**, не flat index:

- `StructAction` (type + args): ROLL, BUY, PLACE, COMPLETE_TURN, MAGNET, APPLY_EFFECT, …
- board order: autoregressive picks → `board_perm` на `COMPLETE_TURN`
- dispatch: `src/training/controller_step.py` — `act_structured` / `step_structured_for_seat`

Flat MLP для BG **запрещён** (`reject_flat_bg_network` в `run_distributed.py`).

---

## Model

### Agent

`MiniBGPPOStructuredAgent` — единый PPO agent для MiniBG и BGLike.  
Rollout buffer: `StructuredMiniBGRolloutBuffer` (хранит structured legal sets в transition info).

### Network registry

`src/models/ppo_policy_factory.py`:

| `network_type` | Module | Игра |
|----------------|--------|------|
| `minibg_structured` | `MiniBGStructuredActorCritic` | MiniBG |
| `bglike_structured` / `bglike_structured_v1` | v1 | BGLike |
| `bglike_structured_v2` | `BGLikeStructuredActorCriticV2` | BGLike (текущие эксперименты) |

Контракт: `StructuredActorCriticProtocol` (`src/models/structured_ac_protocol.py`):

- entity encoder по регионам: own board, shop, hand, enemy, pending;
- cross-region self-attention;
- structured action head (type + slot/target args);
- board order decoder (autoregressive permutation);
- critic на `cache["trunk"]`.

Checkpoint хранит `ppo_network_type` + `ppo_network_kwargs` — architecture id менять нельзя без migration.

---

## League / self-play

### Конфиг

```yaml
train:
  opponent_sampler:
    type: pool
    params:
      self_play:                    # legacy, но работает
        start_episode: 0
        current_self_fraction: 0.4
        past_self_fraction: 0.4
        max_frozen_agents: 20
        save_every: 1000
        frozen_ema_beta: 0.05
      scripted:
        mode: bglike                # или distribution напрямую
        distribution:
          t1_random: 0.25
          t_up_random: 0.25
          structured: 0.25
          random: 0.25
      league:                       # preferred
        rating:
          kind: ema                 # или trueskill
          ema_beta: 0.05
        sampler:
          kind: fractional          # fractional | pfsp_unified | trueskill_quality
          current_self_fraction: 0.4
          past_self_fraction: 0.4
```

`parse_league_settings()` читает `league:` с fallback на legacy `self_play` keys.

### Slot model

| `slot_id` | Тип | Описание |
|-----------|-----|----------|
| `-1` | CURRENT | learner (всегда актуальные веса) |
| `-2, -3, …` | SCRIPTED | meta-slot на bot key (`t1_random`, …) |
| `≥ 0` | FROZEN | checkpoint из pool |

Scripted keys (BGLike): `t1_random`, `t_up_random`, `structured` — см. `default_bot_constructors()` в `src/envs/bglike/heuristic_bots/bots.py`.

MiniBG bots: `t1_random`, `t_up_random` (и др. из minibg heuristic_bots).

### Rating и sampling

**LeagueController** (`src/training/selfplay/league_state.py`):

- `SlotRegistry` — frozen checkpoints (weights bytes + episode metadata)
- `RatingSystem` — EMA pairwise win-rate или TrueSkill
- `PlacementEmaTracker` — rolling placement EMA (окно 20) для diagnostics

**Frozen sampling:** PFSP — веса `(ema_rate + ε)²`, bias к сильным прошлым self.

**GameRecord** (`src/training/selfplay/game_record.py`):

- MiniBG: `minibg_record_from_learner_score(slot, score)` — 2p pairwise
- BGLike: `game_record_for_lobby_end(...)` — все placements → pairwise learner vs каждый opponent seat

Host вызывает `league.submit(record)` → обновление EMA / TrueSkill → snapshot в `sync_league`.

---

## Run directory

| Файл | Содержимое |
|------|------------|
| `config.yaml` | копия конфига |
| `meta.json` | git commit, seed, device, versions |
| `status.json` | step, episode, steps/sec, heartbeat |
| `self_play_frozen.json` | league snapshot: mu/sigma, placement_ema, match_quality |
| `metrics.csv` | PPO scalars (loss, kl, grad_norm, return_mean, …) |
| `checkpoints/` | `dist_*_<step>.pt`, `init.pt`, `*_final.pt` |
| `pid` | PID процесса (удаляется по завершении) |

### Что смотреть при оценке прогресса

**Главный сигнал:** `self_play_frozen.json` → `placement_ema` learner vs frozen/scripted, `match_quality_vs_current` для frozen slots.

**Вторично:** `return_mean` в metrics (растёт при улучшении placement reward).

**Не primary:** avg Q, value loss без placement/winrate context. Высокий `grad_norm` vs `max_grad_norm` — диагностика instability, не успех.

Пример хорошего тренда (dist_ppo_026): learner `placement_ema` ~3.7, frozen recent checkpoints ~4–6, scripted bots ~7–8.

---

## Конфиг: важные параметры

### `game.params`

| Параметр | MiniBG | BGLike |
|----------|--------|--------|
| `seed` | RNG игры | RNG игры |
| `battle_damage_shaping` | scale shaping (0.06 типично) | то же |
| `num_current_seats` | — | seats под learner (1 = один seat, 4 = multi-seat self-play) |

### `agent.params` (structured PPO)

| Параметр | Типичное | Заметка |
|----------|----------|---------|
| `rollout_steps` | 8096 | min steps на worker за раунд |
| `ppo_epochs` | 4 | |
| `minibatch_size` | 512 | |
| `discount_factor` | 1.0 | episodic BG, GAE на segments |
| `entropy_coef` | 0.01 | |
| `max_grad_norm` | 1.0 | clip threshold |

### `train`

| Параметр | Заметка |
|----------|---------|
| `total_steps` | target steps (host counter) |
| `opponent_sampler` | обязателен, type `pool` |
| `callbacks.checkpoint.interval` | частота frozen pool additions |

---

## Mental model

```
YAML config
  → run_distributed()                    # src/training/run_distributed.py
    → DistributedTrainer                 # src/training/distributed_trainer.py
      → N workers:
          make_*_agent_perspective_env()
          league opponent sampling
          structured rollout collection
      → host:
          merge buffers → PPO update
          LeagueController.submit(GameRecord)
          sync_league → workers
```

Три оси эксперимента:

1. **game:** `minibg` (debug/rules) vs `bglike` (target)
2. **model:** `minibg_structured` / `bglike_structured_v2`
3. **league:** scripted mix + frozen pool + rating/sampler kind

---

## Инварианты и типичные ошибки

1. **Structured policy обязательна** — distributed path не использует flat action index для BG.
2. **BGLike reward ≠ MiniBG reward** — placement, не ±1 winrate; сравнивать runs только внутри одного game.id.
3. **League centralised** — только host обновляет рейтинги; workers применяют snapshot.
4. **BGLike league record** — один `GameRecord` на lobby end; segments не пишут отдельные league updates.
5. **`num_current_seats`** — влияет на credit assignment, opponent mix и число learner seats в lobby.
6. **Shaping ≠ terminal reward** — `battle_damage_shaping` через info + shaping_fn, не через `env.step().reward`.
7. **Checkpoint compat** — `ppo_network_type` в checkpoint; v1 ↔ v2 не interchangeable.
8. **Self-play start** — `start_episode` откладывает current/frozen mix; до этого — scripted only.

---

## Связанные файлы

| Тема | Путь |
|------|------|
| Distributed entry | `src/cli/train_distributed.py`, `src/training/run_distributed.py` |
| Trainer loop | `src/training/distributed_trainer.py` |
| BGLike perspective | `src/training/bglike_perspective.py` |
| League | `src/training/selfplay/league_state.py`, `league_sampler.py`, `game_record.py` |
| Structured control | `src/training/controller_step.py` |
| BGLike env | `src/envs/bglike/lobby_env.py` |
| MiniBG env | `src/envs/minibg/env.py`, `ARCHITECTURE.md` |
| Models | `src/models/minibg_structured_ac.py`, `bglike_structured_v2.py` |
| Eval scripts | `scripts/bglike_checkpoint_head_to_head.py`, `scripts/bglike_replay.py` |
