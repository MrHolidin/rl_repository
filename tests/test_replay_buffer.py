"""Tests for standard and prioritized replay buffers."""

import numpy as np

from src.utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


def _make_transition(value: int):
    obs = np.full((2, 3), value, dtype=np.float32)
    next_obs = obs + 1
    action = int(value % 3)
    reward = float(value)
    done = (value % 5) == 0
    legal_mask = np.array([True, False, True], dtype=bool)
    next_legal_mask = np.array([False, True, True], dtype=bool)
    return obs, action, reward, next_obs, done, legal_mask, next_legal_mask


def test_replay_buffer_push_sample_roundtrip():
    buffer = ReplayBuffer(capacity=3)
    for i in range(3):
        buffer.push(*_make_transition(i))

    (
        obs_batch,
        action_batch,
        reward_batch,
        next_obs_batch,
        done_batch,
        legal_mask_batch,
        next_legal_mask_batch,
        indices,
        weights,
    ) = buffer.sample(batch_size=2)

    assert obs_batch.shape == (2, 2, 3)
    assert action_batch.shape == (2,)
    assert reward_batch.shape == (2,)
    assert next_obs_batch.shape == (2, 2, 3)
    assert done_batch.shape == (2,)
    assert legal_mask_batch.shape == (2, 3)
    assert next_legal_mask_batch.shape == (2, 3)
    assert indices is None
    assert weights.shape == (2,)
    assert np.allclose(weights, 1.0)


def test_prioritized_replay_sample_returns_weights_and_indices():
    np.random.seed(0)
    buffer = PrioritizedReplayBuffer(capacity=5)
    for i in range(5):
        buffer.push(*_make_transition(i + 1))

    sample = buffer.sample(batch_size=3)
    assert len(sample) == 9
    (
        obs_batch,
        action_batch,
        reward_batch,
        next_obs_batch,
        done_batch,
        legal_mask_batch,
        next_legal_mask_batch,
        indices,
        weights,
    ) = sample

    assert obs_batch.shape == (3, 2, 3)
    assert action_batch.shape == (3,)
    assert reward_batch.shape == (3,)
    assert next_obs_batch.shape == (3, 2, 3)
    assert done_batch.shape == (3,)
    assert legal_mask_batch.shape == (3, 3)
    assert next_legal_mask_batch.shape == (3, 3)
    assert indices.shape == (3,)
    assert weights.shape == (3,)
    assert np.isclose(weights.max(), 1.0)
    assert np.all(weights > 0.0)


def test_prioritized_replay_update_priorities():
    buffer = PrioritizedReplayBuffer(capacity=4, eps=1e-5)
    for i in range(4):
        buffer.push(*_make_transition(i + 1))

    indices = np.array([0, 2])
    new_priorities = np.array([0.5, 2.0], dtype=np.float32)
    buffer.update_priorities(indices, new_priorities)

    assert np.allclose(buffer.priorities[indices], new_priorities)

