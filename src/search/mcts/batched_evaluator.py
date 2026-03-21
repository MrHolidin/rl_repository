"""Batched neural network evaluator for MCTS."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple, TypeVar

import numpy as np
import torch

from src.games.turn_based_game import Action, TurnBasedGame

S = TypeVar("S")

StateToDictFn = Callable[[S, TurnBasedGame], dict]
ObsBuildFn = Callable[[dict], np.ndarray]


class BatchedNNEvaluator:
    """
    Efficient batched neural network evaluator for MCTS.

    Converts multiple game states to observations and evaluates them
    in a single batched forward pass.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        game: TurnBasedGame,
        state_to_dict_fn: StateToDictFn,
        obs_build_fn: ObsBuildFn,
        num_actions: int,
        device: torch.device,
    ):
        self.network = network
        self.game = game
        self.state_to_dict_fn = state_to_dict_fn
        self.obs_build_fn = obs_build_fn
        self.num_actions = num_actions
        self.device = device

    def __call__(
        self, states: List[S]
    ) -> Tuple[List[Dict[Action, float]], List[float]]:
        if not states:
            return [], []

        obs_list = []
        mask_list = []

        for state in states:
            legal_actions = list(self.game.legal_actions(state))
            legal_mask = np.zeros(self.num_actions, dtype=np.bool_)
            for a in legal_actions:
                legal_mask[a] = True

            state_dict = self.state_to_dict_fn(state, self.game)
            obs_list.append(self.obs_build_fn(state_dict))
            mask_list.append(legal_mask)

        obs_batch = np.stack(obs_list, axis=0)
        mask_batch = np.stack(mask_list, axis=0)

        with torch.inference_mode():
            obs_t = torch.from_numpy(obs_batch).to(self.device, non_blocking=True)
            mask_t = torch.from_numpy(mask_batch).to(self.device, non_blocking=True)
            policy_probs, values = self.network.predict(obs_t, mask_t)
            policy_np = policy_probs.cpu().numpy()
            values_np = values.squeeze(-1).cpu().numpy()

        all_priors = []
        for i, mask in enumerate(mask_list):
            policy_row = policy_np[i]
            all_priors.append({a: policy_row[a] for a in range(self.num_actions) if mask[a]})

        return all_priors, values_np.tolist()


def make_batched_evaluator(
    agent,
    game: TurnBasedGame,
    state_to_dict_fn: StateToDictFn,
    obs_builder,
) -> BatchedNNEvaluator:
    """Create a batched evaluator from an AlphaZero agent."""
    return BatchedNNEvaluator(
        network=agent.network,
        game=game,
        state_to_dict_fn=state_to_dict_fn,
        obs_build_fn=obs_builder.build,
        num_actions=agent.num_actions,
        device=agent.device,
    )
