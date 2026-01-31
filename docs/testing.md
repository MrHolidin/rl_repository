# Testing

## Quick Start

```bash
pytest tests/                           # all tests
pytest tests/ -x                        # stop on first failure
pytest tests/test_checkpoint_compat.py  # only checkpoint tests
```

## Canonical Checkpoint Tests

`tests/test_checkpoint_compat.py` — tests for determinism and model compatibility. **Slower than other tests** (~10 sec) because they involve actual training.

### How It Works

1. **Canonical files** — reference files in `tests/fixtures/`:
   - `canonical_dqn.pt` — checkpoint after 300 training steps with seed=42
   - `canonical_dqn_probe.json` — expected actions on test states

2. **Tests:**
   - `test_checkpoint_produces_same_inference` — loads checkpoint, verifies actions match probe
   - `test_training_deterministic` — trains twice with same seed, actions must match
   - `test_training_matches_canonical` — trains with seed=42, result must match canonical probe

3. **Probe logic:** after training, the agent takes actions on several fixed states. These actions are recorded in probe.json and serve as a fingerprint of the trained model.

### When to Update Canonical Files

Only when **intentionally** changing training behavior (network architecture, training loop logic). Command:

```bash
UPDATE_CANONICAL=1 pytest tests/test_checkpoint_compat.py -k test_update_canonical
```

Then commit `tests/fixtures/canonical_dqn.pt` and `canonical_dqn_probe.json`.

### If Tests Fail

1. **test_training_deterministic** — determinism broken (check seeding, CUDA determinism flags)
2. **test_training_matches_canonical** — behavior changed. If intentional — update canonical. If not — look for regression.
