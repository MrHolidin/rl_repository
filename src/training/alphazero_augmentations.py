"""Data augmentations for AlphaZero training samples."""

from typing import List

import numpy as np

from src.agents.alphazero.replay_buffer import AlphaZeroSample


def horizontal_flip_augment(sample: AlphaZeroSample) -> List[AlphaZeroSample]:
    """Augment an AlphaZero sample by horizontal flip (works for symmetric boards)."""
    return [
        AlphaZeroSample(
            observation=np.flip(sample.observation, axis=-1).copy(),
            legal_mask=np.flip(sample.legal_mask).copy(),
            target_policy=np.flip(sample.target_policy).copy(),
            target_value=sample.target_value,
        )
    ]
