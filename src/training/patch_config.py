"""Resolve active BG patch package for training (env + agent sizing)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from src.bg_catalog.patch_context import PatchContext, load_patch_context


def resolve_patch_context(
    game_params: Optional[Mapping[str, Any]] = None,
    *,
    patch_dir: Optional[str] = None,
) -> PatchContext:
    if patch_dir is not None:
        return load_patch_context(patch_dir)
    if game_params is not None:
        raw = game_params.get("patch_dir")
        if raw:
            return load_patch_context(str(raw))
    raise ValueError(
        "patch_dir is required in game_params (or pass patch_dir= explicitly)"
    )


def _set_agent_param_from_patch(
    agent_params: Dict[str, Any],
    key: str,
    expected: int,
) -> None:
    """Set ``key`` from patch context, or reject a pre-set conflicting value."""
    if key in agent_params:
        actual = agent_params[key]
        if int(actual) != int(expected):
            raise ValueError(
                f"agent_params[{key!r}]={actual!r} != patch context {expected!r}"
            )
        return
    agent_params[key] = expected


def apply_patch_to_agent_params(
    game_params: Dict[str, Any],
    agent_params: Dict[str, Any],
) -> PatchContext:
    """Set ``num_pool_indices`` and ``patch_build`` on agent params from game patch_dir."""
    ctx = resolve_patch_context(game_params)
    _set_agent_param_from_patch(agent_params, "num_pool_indices", ctx.num_pool_indices)
    _set_agent_param_from_patch(agent_params, "patch_build", ctx.build)
    return ctx


def assert_checkpoint_patch_build(
    checkpoint: Mapping[str, Any],
    expected_patch_build: Optional[int],
) -> None:
    """Reject checkpoint reload when ``patch_build`` metadata disagrees with config."""
    if expected_patch_build is None:
        return
    ck_build = checkpoint.get("patch_build")
    if ck_build is None:
        return
    if int(ck_build) != int(expected_patch_build):
        raise ValueError(
            f"checkpoint patch_build {ck_build!r} != expected {expected_patch_build!r}"
        )


__all__ = [
    "apply_patch_to_agent_params",
    "assert_checkpoint_patch_build",
    "resolve_patch_context",
]
