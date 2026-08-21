"""Loading a checkpoint trained before the action space grew.

The action space only ever grows by appending: a new action takes an id above
every id that already existed, and no earlier id changes meaning. That keeps an
old policy *correct* — every action it learned is still the action it learned —
but it does not keep the tensors the same size. A parameter indexed by an
action id, or by an action-type id, has one more row than the checkpoint holds.

So the checkpoint's rows are a prefix of the new parameter's, and loading is a
copy into the first rows with the appended ones left as initialized. On the
patch the policy was trained for, the appended action is never legal, so those
rows are never read and the net behaves exactly as it did.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch.nn as nn

__all__ = ["grow_appended_rows", "current_action_space_size"]


def grow_appended_rows(model: nn.Module, state_dict: Dict[str, Any]) -> List[str]:
    """Pad every checkpoint tensor that is a row-prefix of the model's.

    Mutates ``state_dict`` in place and returns a description of what grew, for
    the caller to log. Anything that differs in any *other* way is left alone,
    so a genuine mismatch still fails the caller's strict check.
    """
    own = model.state_dict()
    grown: List[str] = []
    for key, old in list(state_dict.items()):
        new = own.get(key)
        if new is None or tuple(old.shape) == tuple(new.shape):
            continue
        if old.dim() != new.dim() or tuple(old.shape[1:]) != tuple(new.shape[1:]):
            continue
        if old.shape[0] >= new.shape[0]:
            continue
        merged = new.detach().clone()
        merged[: old.shape[0]] = old.to(merged.dtype)
        state_dict[key] = merged
        grown.append(f"{key} {tuple(old.shape)}->{tuple(new.shape)}")
    return grown


def current_action_space_size(ppo_network_type: str, stored: int) -> int:
    """How wide the env's action space is *now*, for a net of this family.

    A checkpoint records the width it was trained at. The env's legal mask is
    built at today's width, so a head narrower than the mask cannot score it —
    the head grows and ``grow_appended_rows`` fills the rows it already had.
    Never shrinks: a checkpoint wider than the env keeps its own size.
    """
    kind = str(ppo_network_type or "")
    if kind.startswith("minibg"):
        from src.envs.minibg.action_map import NUM_ENV_ACTIONS as width
        from src.envs.minibg.actions import NUM_CORE_ACTIONS as core
    else:
        from src.envs.bglike.action_map import NUM_ENV_ACTIONS as width
        from src.envs.bglike.actions import NUM_CORE_ACTIONS as core
    # Only a checkpoint that already covers the game band is one of this env's.
    # A narrower one is a different net — a test stub, a toy env — and growing
    # it to the lobby's width would be inventing a policy it never had.
    if int(stored) < int(core):
        return int(stored)
    return max(int(stored), int(width))
