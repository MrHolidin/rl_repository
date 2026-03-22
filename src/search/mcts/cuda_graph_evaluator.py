"""CUDA Graph-based neural network evaluator for MCTS.

Uses CUDA Graphs to minimize kernel launch overhead by capturing
the forward pass once and replaying it with new data.
"""

from typing import Callable, Dict, List, Tuple, TypeVar

import numpy as np
import torch
import torch.nn as nn

from src.games.turn_based_game import TurnBasedGame

S = TypeVar("S")
Action = int
StateToDictFn = Callable[[S, "TurnBasedGame"], dict]
ObsBuildFn = Callable[[dict], np.ndarray]


class CUDAGraphEvaluator:
    """
    Batched NN evaluator using CUDA Graphs for reduced kernel launch overhead.
    
    Pads all batches to a fixed size and replays a pre-captured CUDA graph
    instead of launching individual kernels for each forward pass.
    """

    def __init__(
        self,
        network: nn.Module,
        game: TurnBasedGame,
        state_to_dict_fn: StateToDictFn,
        obs_build_fn: ObsBuildFn,
        num_actions: int,
        device: torch.device,
        fixed_batch_size: int = 48,
        direct_forward_threshold: int = 0,
    ):
        self.network = network
        self.game = game
        self.state_to_dict_fn = state_to_dict_fn
        self.obs_build_fn = obs_build_fn
        self.num_actions = num_actions
        self.device = device
        self.fixed_batch_size = fixed_batch_size
        # Batches at or below this size skip the CUDA graph and use a direct
        # inference_mode forward, which is faster for small batches because the
        # CUDA graph pads to fixed_batch_size (often wasting most of the capacity).
        self.direct_forward_threshold = direct_forward_threshold

        # Infer observation shape from a dummy state
        dummy_state = game.initial_state()
        dummy_dict = state_to_dict_fn(dummy_state, game)
        dummy_obs = obs_build_fn(dummy_dict)
        self.obs_shape = dummy_obs.shape

        # Pre-allocate static GPU buffers
        self.static_input = torch.zeros(
            fixed_batch_size, *self.obs_shape, device=device, dtype=torch.float32
        )
        self.static_mask = torch.ones(
            fixed_batch_size, num_actions, device=device, dtype=torch.bool
        )
        self.static_policy = None
        self.static_value = None

        # Capture CUDA graph
        self._capture_graph()

    def _capture_graph(self):
        """Warmup and capture the CUDA graph for forward pass."""
        # Warmup to ensure CUDA is ready
        for _ in range(5):
            with torch.inference_mode():
                self.network.predict(self.static_input, self.static_mask)
        torch.cuda.synchronize()

        # Capture graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with torch.inference_mode():
                self.static_policy, self.static_value = self.network.predict(
                    self.static_input, self.static_mask
                )
        torch.cuda.synchronize()

    def __call__(
        self, states: List[S]
    ) -> Tuple[List[Dict[Action, float]], List[float]]:
        """
        Evaluate states using CUDA graph replay.
        
        Args:
            states: List of game states to evaluate (max fixed_batch_size)
            
        Returns:
            Tuple of (policy_dicts, values) for each state
        """
        n = len(states)
        if n > self.fixed_batch_size:
            raise ValueError(
                f"Batch size {n} exceeds fixed_batch_size {self.fixed_batch_size}"
            )

        # Compute legal actions once per state and pass the pre-built mask to
        # state_to_dict_fn so it does not call game.legal_actions a second time.
        legal_actions_list = []
        if n <= self.direct_forward_threshold:
            obs_np = np.zeros((n, *self.obs_shape), dtype=np.float32)
            mask_np = np.zeros((n, self.num_actions), dtype=np.float32)
            for i, state in enumerate(states):
                legal = list(self.game.legal_actions(state))
                legal_actions_list.append(legal)
                legal_mask_arr = np.zeros(self.num_actions, dtype=bool)
                legal_mask_arr[legal] = True
                state_dict = self.state_to_dict_fn(state, self.game, legal_mask=legal_mask_arr)
                obs_np[i] = self.obs_build_fn(state_dict)
                mask_np[i, legal] = 1

            obs_t = torch.from_numpy(obs_np).to(self.device, non_blocking=True)
            mask_t = torch.from_numpy(mask_np).bool().to(self.device, non_blocking=True)
            with torch.inference_mode():
                policy_t, value_t = self.network.predict(obs_t, mask_t)
            policy_np = policy_t.cpu().numpy()
            values_np = value_t.cpu().numpy()
        else:
            obs_np = np.zeros(
                (self.fixed_batch_size, *self.obs_shape), dtype=np.float32
            )
            mask_np = np.zeros(
                (self.fixed_batch_size, self.num_actions), dtype=np.float32
            )
            for i, state in enumerate(states):
                legal = list(self.game.legal_actions(state))
                legal_actions_list.append(legal)
                legal_mask_arr = np.zeros(self.num_actions, dtype=bool)
                legal_mask_arr[legal] = True
                state_dict = self.state_to_dict_fn(state, self.game, legal_mask=legal_mask_arr)
                obs_np[i] = self.obs_build_fn(state_dict)
                mask_np[i, legal] = 1

            self.static_input.copy_(
                torch.from_numpy(obs_np).to(self.device, non_blocking=True)
            )
            self.static_mask.copy_(
                torch.from_numpy(mask_np).bool().to(self.device, non_blocking=True)
            )
            self.graph.replay()

            policy_np = self.static_policy[:n].cpu().numpy()
            values_np = self.static_value[:n].cpu().numpy()

        result_policies = []
        result_values = []
        for i, legal in enumerate(legal_actions_list):
            probs = policy_np[i]
            policy_dict = {a: float(probs[a]) for a in legal}
            result_policies.append(policy_dict)
            result_values.append(float(values_np[i, 0]))

        return result_policies, result_values


def make_cuda_graph_evaluator(
    agent,
    game: TurnBasedGame,
    state_to_dict_fn: StateToDictFn,
    obs_builder,
    fixed_batch_size: int = 48,
) -> CUDAGraphEvaluator:
    """
    Factory function to create a CUDAGraphEvaluator from an AlphaZero agent.
    
    Args:
        agent: AlphaZeroAgent with network and device
        game: TurnBasedGame instance
        state_to_dict_fn: Function to convert state to dict
        obs_builder: ObservationBuilder with build() method
        fixed_batch_size: Fixed batch size for CUDA graph (pad smaller batches)
        
    Returns:
        CUDAGraphEvaluator instance
    """
    return CUDAGraphEvaluator(
        network=agent.network,
        game=game,
        state_to_dict_fn=state_to_dict_fn,
        obs_build_fn=obs_builder.build,
        num_actions=agent.num_actions,
        device=agent.device,
        fixed_batch_size=fixed_batch_size,
    )
