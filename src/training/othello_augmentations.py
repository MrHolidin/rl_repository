"""Data augmentations for Othello transitions using D4 symmetry group."""

from typing import Callable, List

import numpy as np

from src.training.trainer import Transition


def _transform_action(action: int, size: int, transform: str) -> int:
    """Transform action index according to board transformation."""
    row, col = divmod(action, size)
    
    if transform == "rot90":
        new_row, new_col = col, size - 1 - row
    elif transform == "rot180":
        new_row, new_col = size - 1 - row, size - 1 - col
    elif transform == "rot270":
        new_row, new_col = size - 1 - col, row
    elif transform == "flip_h":
        new_row, new_col = row, size - 1 - col
    elif transform == "flip_v":
        new_row, new_col = size - 1 - row, col
    elif transform == "flip_diag":
        new_row, new_col = col, row
    elif transform == "flip_antidiag":
        new_row, new_col = size - 1 - col, size - 1 - row
    else:
        raise ValueError(f"Unknown transform: {transform}")
    
    return new_row * size + new_col


def _transform_obs(obs: np.ndarray, transform: str) -> np.ndarray:
    """Transform observation (C, H, W) according to board transformation."""
    if transform == "rot90":
        return np.rot90(obs, k=1, axes=(-2, -1)).copy()
    elif transform == "rot180":
        return np.rot90(obs, k=2, axes=(-2, -1)).copy()
    elif transform == "rot270":
        return np.rot90(obs, k=3, axes=(-2, -1)).copy()
    elif transform == "flip_h":
        return np.flip(obs, axis=-1).copy()
    elif transform == "flip_v":
        return np.flip(obs, axis=-2).copy()
    elif transform == "flip_diag":
        return np.swapaxes(obs, -2, -1).copy()
    elif transform == "flip_antidiag":
        return np.flip(np.swapaxes(obs, -2, -1), axis=(-2, -1)).copy()
    else:
        raise ValueError(f"Unknown transform: {transform}")


def _transform_mask(mask: np.ndarray, size: int, transform: str) -> np.ndarray:
    """Transform legal action mask according to board transformation."""
    mask_2d = mask.reshape(size, size)
    
    if transform == "rot90":
        new_mask = np.rot90(mask_2d, k=1)
    elif transform == "rot180":
        new_mask = np.rot90(mask_2d, k=2)
    elif transform == "rot270":
        new_mask = np.rot90(mask_2d, k=3)
    elif transform == "flip_h":
        new_mask = np.flip(mask_2d, axis=1)
    elif transform == "flip_v":
        new_mask = np.flip(mask_2d, axis=0)
    elif transform == "flip_diag":
        new_mask = mask_2d.T
    elif transform == "flip_antidiag":
        new_mask = np.flip(np.flip(mask_2d, axis=0).T, axis=0)
    else:
        raise ValueError(f"Unknown transform: {transform}")
    
    return new_mask.flatten().copy()


def make_othello_d4_augmenter(size: int = 8) -> Callable[[Transition], List[Transition]]:
    """
    Create augmenter using D4 dihedral group symmetries (8-fold).
    
    Returns 7 augmented transitions (all transforms except identity):
    - 3 rotations (90°, 180°, 270°)
    - 2 flips (horizontal, vertical)  
    - 2 diagonal flips (main diagonal, anti-diagonal)
    """
    transforms = ["rot90", "rot180", "rot270", "flip_h", "flip_v", "flip_diag", "flip_antidiag"]
    
    def augment(transition: Transition) -> List[Transition]:
        augmented = []
        
        for transform in transforms:
            new_obs = _transform_obs(transition.obs, transform)
            new_next_obs = _transform_obs(transition.next_obs, transform)
            new_action = _transform_action(int(transition.action), size, transform)
            
            new_legal = (
                _transform_mask(transition.legal_mask, size, transform)
                if transition.legal_mask is not None else None
            )
            new_next_legal = (
                _transform_mask(transition.next_legal_mask, size, transform)
                if transition.next_legal_mask is not None else None
            )
            
            info_copy = transition.info.copy() if isinstance(transition.info, dict) else transition.info
            
            augmented.append(Transition(
                obs=new_obs,
                action=new_action,
                reward=transition.reward,
                next_obs=new_next_obs,
                terminated=transition.terminated,
                truncated=transition.truncated,
                info=info_copy,
                legal_mask=new_legal,
                next_legal_mask=new_next_legal,
            ))
        
        return augmented
    
    return augment


def make_othello_rotation_augmenter(size: int = 8) -> Callable[[Transition], List[Transition]]:
    """
    Create augmenter using only rotations (4-fold, lighter than full D4).
    
    Returns 3 augmented transitions (90°, 180°, 270° rotations).
    """
    transforms = ["rot90", "rot180", "rot270"]
    
    def augment(transition: Transition) -> List[Transition]:
        augmented = []
        
        for transform in transforms:
            new_obs = _transform_obs(transition.obs, transform)
            new_next_obs = _transform_obs(transition.next_obs, transform)
            new_action = _transform_action(int(transition.action), size, transform)
            
            new_legal = (
                _transform_mask(transition.legal_mask, size, transform)
                if transition.legal_mask is not None else None
            )
            new_next_legal = (
                _transform_mask(transition.next_legal_mask, size, transform)
                if transition.next_legal_mask is not None else None
            )
            
            info_copy = transition.info.copy() if isinstance(transition.info, dict) else transition.info
            
            augmented.append(Transition(
                obs=new_obs,
                action=new_action,
                reward=transition.reward,
                next_obs=new_next_obs,
                terminated=transition.terminated,
                truncated=transition.truncated,
                info=info_copy,
                legal_mask=new_legal,
                next_legal_mask=new_next_legal,
            ))
        
        return augmented
    
    return augment
