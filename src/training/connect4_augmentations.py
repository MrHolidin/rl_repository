"""Data augmentations for Connect Four-specific transitions."""

from typing import Callable, List

import numpy as np

from src.training.trainer import Transition


def _flip_obs_horizontally(obs: np.ndarray) -> np.ndarray:
    """
    Flip observation along the last axis (columns axis for BoardChannels).
    """
    if obs.ndim < 2:
        raise ValueError(f"Cannot horizontally flip observation with shape {obs.shape}")
    return np.flip(obs, axis=-1)


def _flip_mask_horizontally(mask: np.ndarray) -> np.ndarray:
    """Reverse column ordering in action masks."""
    mask_arr = np.asarray(mask, dtype=bool)
    return mask_arr[::-1]


def make_connect4_horizontal_augmenter(num_cols: int) -> Callable[[Transition], List[Transition]]:
    """
    Create an augmenter that mirrors Connect Four transitions horizontally.

    The augmenter flips both observations, maps the agent action to its mirrored
    column, and reverses legal-action masks.
    """

    def augment(transition: Transition) -> List[Transition]:
        flipped_obs = _flip_obs_horizontally(transition.obs)
        flipped_next_obs = _flip_obs_horizontally(transition.next_obs)
        flipped_action = num_cols - 1 - int(transition.action)

        flipped_legal = (
            _flip_mask_horizontally(transition.legal_mask) if transition.legal_mask is not None else None
        )
        flipped_next_legal = (
            _flip_mask_horizontally(transition.next_legal_mask) if transition.next_legal_mask is not None else None
        )

        info_copy = transition.info.copy() if isinstance(transition.info, dict) else transition.info

        flipped_transition = Transition(
            obs=flipped_obs,
            action=flipped_action,
            reward=transition.reward,
            next_obs=flipped_next_obs,
            terminated=transition.terminated,
            truncated=transition.truncated,
            info=info_copy,
            legal_mask=flipped_legal,
            next_legal_mask=flipped_next_legal,
        )

        return [flipped_transition]

    return augment

