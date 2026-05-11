"""Agent modules."""

from .base_agent import BaseAgent
from .random_agent import RandomAgent
from .connect4 import HeuristicAgent, SmartHeuristicAgent
from .othello import OthelloHeuristicAgent
from .qlearning_agent import QLearningAgent
from .dqn.agent import DQNAgent
from .ppo_agent import PPOAgent
from .alphazero.agent import AlphaZeroAgent
from ..features.action_space import DiscreteActionSpace
from ..features.observation_builder import BoardChannels
from ..models import Connect4DQN, Connect4QRDQN, OthelloDQN, OthelloQRDQN
from ..models.ppo_policy_factory import (
    PPO_NETWORK_MINIBG_SLOT,
    build_ppo_actor_critic,
    default_ppo_network_kwargs,
    ppo_network_type_for_save,
)
from ..registry import list_agents, register_agent

_PPO_AGENT_KWARGS = frozenset(
    {
        "observation_shape",
        "observation_type",
        "num_actions",
        "network",
        "ppo_network_type",
        "ppo_network_kwargs",
        "learning_rate",
        "discount_factor",
        "gae_lambda",
        "ppo_clip_eps",
        "clip_value_loss",
        "value_clip_eps",
        "entropy_coef",
        "value_coef",
        "max_grad_norm",
        "rollout_steps",
        "ppo_epochs",
        "minibatch_size",
        "device",
        "seed",
        "model_config",
        "action_space",
        "compute_detailed_metrics",
    }
)


def _filter_ppo_agent_kwargs(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if k in _PPO_AGENT_KWARGS}


if "random" not in list_agents():
    register_agent("random", RandomAgent)
if "heuristic" not in list_agents():
    register_agent("heuristic", HeuristicAgent)
if "smart_heuristic" not in list_agents():
    register_agent("smart_heuristic", SmartHeuristicAgent)
if "othello_heuristic" not in list_agents():
    register_agent("othello_heuristic", OthelloHeuristicAgent)
if "qlearning" not in list_agents():
    register_agent("qlearning", QLearningAgent)
if "dqn" not in list_agents():
    def _dqn_factory(**kwargs):
        # If network is already provided, use it directly
        if "network" in kwargs:
            kwargs.pop("observation_shape", None)
            kwargs.pop("observation_type", None)
            kwargs.pop("network_type", None)
            kwargs.pop("dueling", None)
            return DQNAgent(**kwargs)
        
        # Extract network-related params
        obs_shape = kwargs.pop("observation_shape", None)
        kwargs.pop("observation_type", None)  # not used anymore
        network_type = kwargs.pop("network_type", "dqn")
        
        action_space = kwargs.get("action_space")
        num_actions = kwargs.get("num_actions")

        if action_space is None and num_actions is not None:
            action_space = DiscreteActionSpace(num_actions)
            kwargs["action_space"] = action_space

        # Default to Connect4 dimensions
        if obs_shape is None:
            builder = BoardChannels(board_shape=(6, 7))
            obs_shape = builder.observation_shape
        
        if num_actions is None:
            num_actions = 7
            kwargs["num_actions"] = num_actions
        
        if action_space is None:
            action_space = DiscreteActionSpace(num_actions)
            kwargs["action_space"] = action_space

        if network_type == "minibg_mlp":
            from ..models.simple_mlp import SimpleMLP

            if obs_shape is None or len(obs_shape) != 1:
                raise ValueError(
                    "network_type 'minibg_mlp' requires agent.params.observation_shape: [D] "
                    "(flat vector, currently [362] for MiniBG; see src/envs/minibg/obs.py:OBS_DIM)."
                )
            dueling = kwargs.pop("dueling", None)
            if dueling is None:
                dueling = True
            hidden_size = int(kwargs.pop("mlp_hidden_size", 256))
            network = SimpleMLP(
                input_size=int(obs_shape[0]),
                num_actions=num_actions,
                hidden_size=hidden_size,
                dueling=dueling,
            )
            return DQNAgent(network=network, **kwargs)

        if network_type == "minibg_slot":
            from ..models.minibg_slot_net import MiniBGSlotEncoderNet, _OBS_DIM

            if obs_shape is None or len(obs_shape) != 1 or int(obs_shape[0]) != _OBS_DIM:
                raise ValueError(
                    f"network_type 'minibg_slot' requires observation_shape [{_OBS_DIM}] (MiniBG flat obs)."
                )
            dueling = kwargs.pop("dueling", None)
            if dueling is None:
                dueling = True
            slot_hidden = int(kwargs.pop("slot_hidden_channels", 64))
            trunk_hidden = int(kwargs.pop("trunk_hidden_size", 256))
            use_noisy_nets = bool(kwargs.get("use_noisy_nets", False))
            noisy_sigma = float(kwargs.pop("noisy_sigma", 0.5))
            network = MiniBGSlotEncoderNet(
                num_actions=num_actions,
                slot_hidden=slot_hidden,
                trunk_hidden=trunk_hidden,
                dueling=dueling,
                use_noisy=use_noisy_nets,
                noisy_sigma=noisy_sigma,
            )
            return DQNAgent(network=network, **kwargs)

        # Create network based on board size
        in_channels = obs_shape[0] if len(obs_shape) == 3 else 3
        rows = obs_shape[1] if len(obs_shape) == 3 else 6
        cols = obs_shape[2] if len(obs_shape) == 3 else 7
        dueling = kwargs.pop("dueling", None)
        if dueling is None:
            dueling = network_type == "dueling_dqn"
        use_distributional = kwargs.pop("use_distributional", False)
        n_quantiles = kwargs.pop("n_quantiles", 32)
        # Noisy nets: agent's use_noisy_nets controls both network's use_noisy and agent behavior
        use_noisy_nets = kwargs.get("use_noisy_nets", False)
        noisy_sigma = kwargs.pop("noisy_sigma", 0.5)
        
        if use_distributional:
            if rows == 8 and cols == 8 and num_actions == 64:
                network = OthelloQRDQN(
                    board_size=8,
                    in_channels=in_channels,
                    num_actions=num_actions,
                    n_quantiles=n_quantiles,
                    dueling=dueling,
                    use_noisy=use_noisy_nets,
                    noisy_sigma=noisy_sigma,
                )
            else:
                network = Connect4QRDQN(
                    rows=rows,
                    cols=cols,
                    in_channels=in_channels,
                    num_actions=num_actions,
                    n_quantiles=n_quantiles,
                    dueling=dueling,
                    use_noisy=use_noisy_nets,
                    noisy_sigma=noisy_sigma,
                )
            kwargs["use_distributional"] = True
            kwargs["n_quantiles"] = n_quantiles
        else:
            if rows == 8 and cols == 8 and num_actions == 64:
                network = OthelloDQN(
                    board_size=8,
                    in_channels=in_channels,
                    num_actions=num_actions,
                    dueling=dueling,
                    use_noisy=use_noisy_nets,
                    noisy_sigma=noisy_sigma,
                )
            else:
                network = Connect4DQN(
                    rows=rows,
                    cols=cols,
                    in_channels=in_channels,
                    num_actions=num_actions,
                    dueling=dueling,
                    use_noisy=use_noisy_nets,
                    noisy_sigma=noisy_sigma,
                )
        
        return DQNAgent(network=network, **kwargs)
    register_agent("dqn", _dqn_factory)
if "ppo" not in list_agents():
    def _ppo_factory(**kwargs):
        if kwargs.get("network") is not None:
            kwargs.setdefault("ppo_network_type", "actor_critic_cnn")
            if kwargs.get("ppo_network_kwargs") is None:
                kwargs["ppo_network_kwargs"] = default_ppo_network_kwargs(
                    str(kwargs["ppo_network_type"]),
                    kwargs["network"],
                )
            return PPOAgent(**_filter_ppo_agent_kwargs(kwargs))

        network_type = str(kwargs.pop("network_type", "board_cnn")).strip().lower()
        action_space = kwargs.get("action_space")
        num_actions = kwargs.get("num_actions")

        if action_space is None and num_actions is not None:
            kwargs["action_space"] = DiscreteActionSpace(num_actions)
            action_space = kwargs["action_space"]

        is_minibg = network_type == PPO_NETWORK_MINIBG_SLOT
        obs_shape = kwargs.get("observation_shape")
        obs_type = kwargs.get("observation_type")

        if is_minibg:
            if obs_shape is None or num_actions is None:
                raise ValueError(
                    "PPO network_type minibg_slot requires observation_shape (e.g. [362]) "
                    "and num_actions from the environment config."
                )
            slot_hidden_channels = int(kwargs.pop("slot_hidden_channels", 16))
            trunk_hidden_size = int(kwargs.pop("trunk_hidden_size", 256))
            region_conv2_kernel = int(kwargs.pop("region_conv2_kernel", 1))
            net = build_ppo_actor_critic(
                PPO_NETWORK_MINIBG_SLOT,
                tuple(obs_shape),
                int(num_actions),
                slot_hidden_channels=slot_hidden_channels,
                trunk_hidden_size=trunk_hidden_size,
                region_conv2_kernel=region_conv2_kernel,
            )
            kwargs["observation_type"] = obs_type or "vector"
            ppo_kw = {
                "slot_hidden": slot_hidden_channels,
                "trunk_hidden": trunk_hidden_size,
                "region_conv2_kernel": region_conv2_kernel,
            }
        else:
            if obs_shape is None or obs_type is None or num_actions is None:
                builder = BoardChannels(board_shape=(6, 7))
                default_action_space = action_space or DiscreteActionSpace(n=7)
                kwargs.setdefault("observation_shape", builder.observation_shape)
                kwargs.setdefault("observation_type", builder.observation_type)
                kwargs.setdefault("action_space", default_action_space)
                kwargs.setdefault("num_actions", default_action_space.size)
            else:
                kwargs.setdefault("observation_shape", obs_shape)
                kwargs.setdefault("observation_type", obs_type)
                if action_space is None:
                    kwargs.setdefault("action_space", DiscreteActionSpace(num_actions))
                kwargs.setdefault("num_actions", num_actions)

            obs_shape = kwargs["observation_shape"]
            num_actions = kwargs["num_actions"]
            if len(obs_shape) != 3:
                builder = BoardChannels(board_shape=(6, 7))
                kwargs["observation_shape"] = builder.observation_shape
                kwargs.setdefault("observation_type", builder.observation_type)
                obs_shape = kwargs["observation_shape"]
            net = build_ppo_actor_critic(
                network_type,
                tuple(obs_shape),
                int(num_actions),
            )
            ppo_kw = default_ppo_network_kwargs(
                ppo_network_type_for_save(network_type),
                net,
            )

        kwargs["network"] = net
        kwargs["ppo_network_type"] = ppo_network_type_for_save(network_type)
        kwargs["ppo_network_kwargs"] = ppo_kw
        return PPOAgent(**_filter_ppo_agent_kwargs(kwargs))

    register_agent("ppo", _ppo_factory)

__all__ = [
    "BaseAgent",
    "RandomAgent",
    "HeuristicAgent",
    "SmartHeuristicAgent",
    "QLearningAgent",
    "DQNAgent",
    "PPOAgent",
    "AlphaZeroAgent",
]
