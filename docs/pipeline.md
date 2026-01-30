# Training Pipeline

Single entrypoint for running training from a YAML config: `python -m src.cli.train`. No Hydra; config is loaded from a single file, and each run writes to a **run directory** (`run_dir`) with a copy of the config, metadata, and checkpoints.

## Quick start

```bash
python -m src.cli.train --config configs/dqn_connect4.yaml
```

Optional `--run_dir`:

```bash
python -m src.cli.train --config configs/dqn_connect4.yaml --run_dir runs/my_run
```

If `--run_dir` is omitted, it defaults to `runs/<config_stem>_<UTC timestamp>`, e.g. `runs/dqn_connect4_2026-01-29_12-00-00`.

## Run directory layout

Every run creates a directory (either given by `--run_dir` or the default) with:

| Path | Description |
|------|-------------|
| `config.yaml` | Copy of the config file used for this run |
| `meta.json` | Run metadata (git, command, seed, device, versions, etc.) |
| `status.json` | Live progress: step, episode, epsilon, heartbeat (updated every 100 steps) |
| `pid` | Process ID (exists while running, removed on completion) |
| `checkpoints/` | Saved model checkpoints (e.g. `model_200.pt`, `model_400.pt`) |

## Managing runs

### List running pipelines

```bash
python -m src.cli.runs list
```

Example output:
```
  my_experiment                            running    step=12345/50000 (24.7%)  ep=678    ε=0.4200  [1h 23m 45s]  (pid=12345)
```

Use `--all` to also show completed/stopped runs:
```bash
python -m src.cli.runs list --all
```

### Stop a running pipeline

```bash
python -m src.cli.runs stop runs/my_experiment
```

This sends `SIGTERM` to the process (instant graceful shutdown). If the signal fails, it creates a `stop` file that the pipeline checks periodically.

### Status file (`status.json`)

Updated every 100 steps (configurable via `status_interval` in `run()`). Fields:

| Field | Description |
|-------|-------------|
| `status` | `running`, `completed`, `stopped`, `dead`, or `stale` |
| `step` | Current training step |
| `total_steps` | Target total steps from config |
| `episode` | Current episode index |
| `epsilon` | Current exploration epsilon (if agent has one) |
| `start_time` | Run start time (ISO format) |
| `last_heartbeat` | Last status update time (ISO format) |

A run is considered **stale** if `last_heartbeat` is older than 2 minutes and **dead** if the PID file exists but the process is not running.

## Config structure (YAML)

Config is a single YAML file. Top-level keys:

| Key | Required | Description |
|-----|----------|-------------|
| `seed` | No | Global seed; if set, RNG is fixed and stored in `meta.json` |
| `game` | Yes | Game/environment id and params |
| `agent` | Yes | Agent id and params (e.g. `dqn` with `learning_rate`, `epsilon`, …) |
| `train` | Yes | Training options and callbacks |
| `eval` | No | Eval options (currently unused by the pipeline; reserved for future eval callback) |

### `game`

- `id`: registered game id (e.g. `connect4`).
- `params`: passed to the game constructor (e.g. `rows`, `cols` for Connect4).

### `agent`

- `id`: registered agent id (e.g. `dqn`).
- `params`: agent-specific (learning rate, epsilon, replay buffer, device, etc.).  
  `device` can be `null` for auto (CUDA if available, else CPU).  
  `num_actions`, `action_space`, `observation_shape` are inferred from the environment when omitted.

### `train`

- `total_steps`: number of training steps (default `10000`).
- `deterministic`: use deterministic policy during training (default `false`).
- `track_timings`: record and print timing breakdown (default `false`).
- `start_policy`: who moves first each episode — `random`, `agent_first`, or `opponent_first` (default `random`).
- **`opponent_sampler`** (required): how opponents are chosen.
- `callbacks`: list of callback configs (see below).

### `train.opponent_sampler`

Must be present. Supported types:

- **`random`**: always use a `RandomAgent`.  
  - `params`: optional `seed`; if missing, uses top-level `seed`.
- **`pool`**: self-play style pool (heuristics + frozen checkpoints).  
  - Requires top-level `seed`.  
  - `params.self_play`: `start_episode`, `current_self_fraction`, `past_self_fraction`, `max_frozen_agents`, `save_every`.  
  - `params.heuristic_distribution`: e.g. `{ "random": 0.2, "heuristic": 0.5, "smart_heuristic": 0.3 }`.

### `train.callbacks`

List of `{ type, enabled?, params? }`. The pipeline always adds a **checkpoint** callback; its `interval` and `prefix` can be overridden via a `checkpoint` entry. Other supported types:

| Type | Description | Common `params` |
|------|-------------|------------------|
| `checkpoint` | Save agent every N steps to `run_dir/checkpoints/` | `interval`, `prefix` |
| `epsilon_decay` | Decay exploration ε each step or episode (agents with `epsilon` / `epsilon_decay` / `epsilon_min`) | `every`: `step` or `episode` |
| `lr_decay` | Multiply optimizer LR by `decay_factor` every N steps | `interval_steps`, `decay_factor`, `min_lr?`, `optimizer_attr`, `metric_key` |

Example:

```yaml
train:
  callbacks:
    - type: checkpoint
      params: { interval: 2000, prefix: model }
    - type: epsilon_decay
      params: { every: step }
    - type: lr_decay
      params:
        interval_steps: 5000
        decay_factor: 0.5
        min_lr: 1.0e-6
        optimizer_attr: optimizer
        metric_key: learning_rate
```

## Metadata (`meta.json`)

Written once per run into `run_dir/meta.json`. Fields:

| Field | Description |
|-------|-------------|
| `git_commit` | `git rev-parse HEAD` at run start (or `null`) |
| `git_dirty` | Whether the working tree had uncommitted changes |
| `command` | Invocation string (e.g. `python -m src.cli.train --config ...`) |
| `start_time_iso` | Run start time (UTC, ISO format) |
| `seed` | Config `seed` (or `null`) |
| `device` | Resolved device (`cuda` or `cpu`) |
| `python_version` | Python version string |
| `torch_version` | `torch.__version__` (or `null` if not installed) |
| `config_path` | Absolute path to the config file used |

## Example config

See `configs/dqn_connect4.yaml` for a full example: Connect4, DQN, `random` opponent sampler, checkpoint + epsilon_decay + lr_decay callbacks.

## Programmatic use

To run the same pipeline from code:

```python
from pathlib import Path
from src.training.run import run

run(
    config_path=Path("configs/dqn_connect4.yaml"),
    run_dir=Path("runs/my_run"),
    command=None,  # optional; defaults to sys.argv
)
```

This creates `run_dir`, writes `config.yaml` and `meta.json`, builds env/agent/callbacks/opponent_sampler from config, and calls `Trainer.train()`.

## See also

- [Agents](agents/README.md) and [invariants](agents/invariants.md) for agent contracts and observation/reward conventions.
- `src.training.trainer` for the training loop and callback hooks.
