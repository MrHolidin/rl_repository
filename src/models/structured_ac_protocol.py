"""Protocol for structured actor-critic networks used by ``MiniBGPPOStructuredAgent``.

Any ``nn.Module`` that satisfies this contract can be plugged into the structured
PPO agent. Distinct versions of the architecture (v1, v2, …) implement the same
surface but can differ internally — letting us train/load several architectures
side by side and compare them via the same agent code.

Cache contract for ``encode_state`` / ``policy_logits_and_value`` /
``policy_logits_value_from_tokens``: returned dict must contain at least
  - ``"trunk"``   — feature tensor consumed by the critic (the head ``self.critic``)
  - ``"g_full"``  — concatenated globals (passed to action / order scoring)
  - ``"E_own"``, ``"E_shop"``, ``"E_hand"``, ``"E_pending"``
                   — per-region entity embeddings ``(B, L_*, slot_hidden)``
  - ``"EXT_own"``, ``"EXT_shop"``, ``"EXT_hand"``, ``"EXT_pending"``
                   — raw trigger+effect bits used by ``ent_extras`` bypass
  - ``"E_enemy"`` (may be empty ``(B, 0, slot_hidden)`` for bglike layout)

Additional keys are allowed. Concrete implementations also expose
``self.critic`` (a Module mapping ``cache["trunk"] → (B, 1)``) and
``self.board_size: int``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple, Union, runtime_checkable

import torch


@runtime_checkable
class StructuredActorCriticProtocol(Protocol):
    """Duck-typed surface that ``MiniBGPPOStructuredAgent`` requires.

    ``runtime_checkable`` checks for method **presence** at isinstance() time —
    signatures and types are not checked. We deliberately do **not** declare the
    ``board_size: int`` and ``critic: nn.Module`` attributes here: ``nn.Module``
    stores submodules in ``_modules`` and exposes them via ``__getattr__``, but
    Python's Protocol ``__instancecheck__`` uses static lookup that bypasses
    ``__getattr__`` — declaring them as Protocol attributes makes every
    ``nn.Module``-based implementation fail isinstance(). Required attributes
    (``board_size``, ``critic``) are documented in this module's docstring; the
    agent fails with a clear error if they're missing at first use.
    """

    def encode_state(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ...

    def policy_logits_and_value(
        self,
        obs: torch.Tensor,
        legal_actions: List[List[Any]],
        *,
        return_cache: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        ...

    def policy_logits_value_from_tokens(
        self,
        obs: torch.Tensor,
        type_ids: torch.Tensor,
        role_ids: torch.Tensor,
        src_region_kinds: torch.Tensor,
        src_region_slots: torch.Tensor,
        tgt_region_kinds: torch.Tensor,
        tgt_region_slots: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_cache: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        ...

    def sample_board_order(
        self,
        state_emb: torch.Tensor,
        e_own: torch.Tensor,
        g_full: torch.Tensor,
        occupied_mask: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

    def order_logprob_given_sequence(
        self,
        state_emb: torch.Tensor,
        e_own: torch.Tensor,
        g_full: torch.Tensor,
        occupied_mask: torch.Tensor,
        picked_slots: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def get_constructor_kwargs(self) -> Dict[str, Any]:
        ...


__all__ = ["StructuredActorCriticProtocol"]
