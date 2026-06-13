"""Agent modules."""

from .base_agent import BaseAgent
from .random_agent import RandomAgent
from .connect4 import HeuristicAgent, SmartHeuristicAgent
from .othello import OthelloHeuristicAgent
from .qlearning_agent import QLearningAgent
from .dqn.agent import DQNAgent
from .ppo_agent import PPOAgent
from .ppo_structured_minibg_agent import MiniBGPPOStructuredAgent
from .ppo_dvd_agent import PPODvDAgent
from .alphazero.agent import AlphaZeroAgent
from ..features.action_space import DiscreteActionSpace
from ..features.observation_builder import BoardChannels
from ..models import Connect4DQN, Connect4QRDQN, OthelloDQN, OthelloQRDQN
from ..models.minibg_structured_ac import MiniBGStructuredActorCritic
from ..models.ppo_policy_factory import (
    PPO_NETWORK_BGLIKE_STRUCTURED,
    PPO_NETWORK_BGLIKE_STRUCTURED_V2,
    PPO_NETWORK_BGLIKE_STRUCTURED_V3,
    PPO_NETWORK_BGLIKE_STRUCTURED_V4,
    PPO_NETWORK_BGLIKE_STRUCTURED_V5,
    PPO_NETWORK_BGLIKE_STRUCTURED_V6,
    PPO_NETWORK_BGLIKE_STRUCTURED_V7,
    PPO_NETWORK_BGLIKE_STRUCTURED_V8,
    PPO_NETWORK_BGLIKE_STRUCTURED_V9,
    PPO_NETWORK_MINIBG_SLOT,
    PPO_NETWORK_MINIBG_STRUCTURED,
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
        "patch_build",
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

        # Default to Connect4 dimensions, or Battlegrounds flat obs when applicable.
        if obs_shape is None:
            if network_type in ("minibg_mlp", "minibg_slot"):
                from src.envs.minibg.obs import OBS_DIM

                obs_shape = (OBS_DIM,)
            else:
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
                    "network_type 'minibg_mlp' requires a 1-D observation vector "
                    "(inferred from src.envs.minibg.obs.OBS_DIM at train startup)."
                )
            dueling = kwargs.pop("dueling", None)
            if dueling is None:
                dueling = True
            hidden_size = int(kwargs.pop("mlp_hidden_size", 256))
            kwargs.pop("noisy_sigma", None)
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
                    f"network_type 'minibg_slot' requires observation dim {_OBS_DIM} "
                    "(inferred from src.envs.minibg.obs.OBS_DIM at train startup)."
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
        is_minibg_structured = network_type == PPO_NETWORK_MINIBG_STRUCTURED
        is_bglike_structured = network_type == PPO_NETWORK_BGLIKE_STRUCTURED
        is_bglike_structured_v2 = network_type == PPO_NETWORK_BGLIKE_STRUCTURED_V2
        is_bglike_structured_v3 = network_type == PPO_NETWORK_BGLIKE_STRUCTURED_V3
        is_bglike_structured_v4 = network_type == PPO_NETWORK_BGLIKE_STRUCTURED_V4
        is_bglike_structured_v5 = network_type == PPO_NETWORK_BGLIKE_STRUCTURED_V5
        is_bglike_structured_v6 = network_type == PPO_NETWORK_BGLIKE_STRUCTURED_V6
        # v8 = v7 + distributional placement critic; v9 = v8 + economy-coloured
        # action queries. Identical kwarg surface and agent (PPODvDAgent), so
        # both ride the v7 branch with a class switch.
        is_bglike_structured_v8 = network_type == PPO_NETWORK_BGLIKE_STRUCTURED_V8
        is_bglike_structured_v9 = network_type == PPO_NETWORK_BGLIKE_STRUCTURED_V9
        is_bglike_structured_v7 = (
            network_type == PPO_NETWORK_BGLIKE_STRUCTURED_V7
            or is_bglike_structured_v8
            or is_bglike_structured_v9
        )
        # v7 shares v6's obs_v5 layout (the env emits OBS_DIM_V5; the DvD agent
        # appends the identity tail before feeding the net).
        is_bglike_v5_or_v6 = (
            is_bglike_structured_v5 or is_bglike_structured_v6 or is_bglike_structured_v7
        )
        is_bglike_v_any = (
            is_bglike_structured
            or is_bglike_structured_v2
            or is_bglike_structured_v3
            or is_bglike_structured_v4
            or is_bglike_structured_v5
            or is_bglike_structured_v6
            or is_bglike_structured_v7
        )
        is_structured_v1 = is_minibg_structured or is_bglike_structured
        is_structured = (
            is_structured_v1
            or is_bglike_structured_v2
            or is_bglike_structured_v3
            or is_bglike_structured_v4
            or is_bglike_structured_v5
            or is_bglike_structured_v6
            or is_bglike_structured_v7
        )
        is_flat_mlp = network_type in ("minibg_mlp", "mlp", "flat_mlp")
        obs_shape = kwargs.get("observation_shape")
        obs_type = kwargs.get("observation_type")
        num_pool_indices = kwargs.pop("num_pool_indices", None)
        patch_build = kwargs.pop("patch_build", None)

        if obs_shape is None:
            if is_bglike_v5_or_v6:
                from src.envs.bglike.obs_v5 import OBS_DIM_V5 as _expected_obs

                obs_shape = (_expected_obs,)
            elif is_bglike_v_any:
                from src.envs.bglike.obs import OBS_DIM as _expected_obs

                obs_shape = (_expected_obs,)
            elif is_minibg or is_minibg_structured:
                from src.envs.minibg.obs import OBS_DIM as _expected_obs

                obs_shape = (_expected_obs,)

        if is_flat_mlp:
            if obs_shape is None or num_actions is None:
                raise ValueError(
                    "PPO network_type minibg_mlp/flat_mlp requires num_actions and "
                    "a 1-D observation vector (inferred at train startup for Battlegrounds)."
                )
            hidden_size = int(kwargs.pop("mlp_hidden_size", 256))
            net = build_ppo_actor_critic(
                "flat_mlp",
                tuple(obs_shape),
                int(num_actions),
                mlp_hidden_size=hidden_size,
            )
            kwargs["observation_type"] = obs_type or "vector"
            ppo_kw = dict(net.get_constructor_kwargs())
            kwargs["network"] = net
            kwargs["ppo_network_type"] = ppo_network_type_for_save("flat_mlp")
            kwargs["ppo_network_kwargs"] = ppo_kw
            kwargs.pop("action_space", None)
            return PPOAgent(**_filter_ppo_agent_kwargs(kwargs))

        if is_structured:
            if obs_shape is None or num_actions is None:
                raise ValueError(
                    f"PPO network_type {network_type!r} requires num_actions and "
                    "a 1-D observation vector (inferred at train startup for Battlegrounds)."
                )
            if is_bglike_v5_or_v6:
                from src.envs.bglike.obs_v5 import OBS_DIM_V5 as _expected_obs

                if tuple(obs_shape) != (_expected_obs,):
                    raise ValueError(
                        f"PPO network_type {network_type!r} requires observation_shape [{_expected_obs}]"
                    )
            elif is_bglike_v_any:
                from src.envs.bglike.obs import OBS_DIM as _expected_obs

                if tuple(obs_shape) != (_expected_obs,):
                    raise ValueError(
                        f"PPO network_type {network_type!r} requires observation_shape [{_expected_obs}]"
                    )

            # Common knobs (defaults differ between v1 and v2 — see below).
            region_conv2_kernel = int(kwargs.pop("region_conv2_kernel", 1))
            action_dim = int(kwargs.pop("action_dim", 64))
            interaction_dim = int(kwargs.pop("interaction_dim", 64))
            order_hidden = int(kwargs.pop("order_hidden", 64))
            order_pos_dim = int(kwargs.pop("order_pos_dim", 16))
            score_hidden = int(kwargs.pop("score_hidden", 128))
            order_score_hidden = int(kwargs.pop("order_score_hidden", 64))
            critic_hidden = int(kwargs.pop("critic_hidden", 128))
            card_emb_dim = int(kwargs.pop("card_emb_dim", 16))
            entity_attention_heads = int(kwargs.pop("entity_attention_heads", 4))
            entity_attention_ff_mult = int(kwargs.pop("entity_attention_ff_mult", 2))
            entity_attention_init_scale = float(kwargs.pop("entity_attention_init_scale", 0.1))

            if (
                is_bglike_structured_v2
                or is_bglike_structured_v3
                or is_bglike_structured_v4
                or is_bglike_structured_v5
                or is_bglike_structured_v6
                or is_bglike_structured_v7
            ):
                # v2 / v3 / v4 / v5 / v6 share the exact same kwarg surface up
                # to and including entity self-attention. v3+ additionally
                # consumes action_cross_attn_* knobs; v4 also consumes
                # recurrent_hidden_dim / round_gru_init_scale; v5 also consumes
                # ability_emb_dim / ability_attention_init_scale; v6 only
                # consumes ability_emb_dim (no separate attention init since
                # the pool is a single-query softmax, not a sub-block).
                # Defaults match the class defaults.
                slot_hidden_channels = int(kwargs.pop("slot_hidden_channels", 48))
                state_dim = int(kwargs.pop("state_dim", 128))
                entity_attention_layers = int(kwargs.pop("entity_attention_layers", 2))
                # v2/v3 ignore trunk_hidden / use_global_entity_token by design;
                # accept-and-drop if a v1 config bleeds through.
                kwargs.pop("trunk_hidden_size", None)
                kwargs.pop("use_global_entity_token", None)
                # Auxiliary battle-prediction head: optional dict in yaml under
                # ``agent.params.battle_pred``. When ``enabled=true`` the model
                # adds a small head + the agent computes Huber loss vs the
                # signed uncapped damage label collected on combat-resolution
                # steps. ``aux_coef`` controls joint backbone-regularization
                # strength; ``detach_features=true`` makes it purely diagnostic.
                battle_pred_config = kwargs.pop("battle_pred", None)
                common_kwargs = dict(
                    slot_hidden=slot_hidden_channels,
                    region_conv2_kernel=region_conv2_kernel,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    interaction_dim=interaction_dim,
                    order_hidden=order_hidden,
                    order_pos_dim=order_pos_dim,
                    score_hidden=score_hidden,
                    order_score_hidden=order_score_hidden,
                    critic_hidden=critic_hidden,
                    card_emb_dim=card_emb_dim,
                    entity_attention_layers=entity_attention_layers,
                    entity_attention_heads=entity_attention_heads,
                    entity_attention_ff_mult=entity_attention_ff_mult,
                    entity_attention_init_scale=entity_attention_init_scale,
                    obs_layout="bglike",
                    num_pool_indices=num_pool_indices,
                    battle_pred_config=battle_pred_config,
                )
                if (
                    is_bglike_structured_v3
                    or is_bglike_structured_v4
                    or is_bglike_structured_v5
                    or is_bglike_structured_v6
                    or is_bglike_structured_v7
                ):
                    action_cross_attn_heads = int(
                        kwargs.pop("action_cross_attn_heads", 4)
                    )
                    action_cross_attn_ff_mult = int(
                        kwargs.pop("action_cross_attn_ff_mult", 2)
                    )
                    action_cross_attn_init_scale = float(
                        kwargs.pop("action_cross_attn_init_scale", 0.1)
                    )
                    v3_v4_kwargs = dict(
                        action_cross_attn_heads=action_cross_attn_heads,
                        action_cross_attn_ff_mult=action_cross_attn_ff_mult,
                        action_cross_attn_init_scale=action_cross_attn_init_scale,
                        **common_kwargs,
                    )
                    if is_bglike_structured_v4:
                        from ..models.bglike_structured_v4 import BGLikeStructuredV4

                        recurrent_hidden_dim = int(
                            kwargs.pop("recurrent_hidden_dim", 128)
                        )
                        round_gru_init_scale = float(
                            kwargs.pop("round_gru_init_scale", 0.1)
                        )
                        net = BGLikeStructuredV4(
                            recurrent_hidden_dim=recurrent_hidden_dim,
                            round_gru_init_scale=round_gru_init_scale,
                            **v3_v4_kwargs,
                        )
                    elif is_bglike_structured_v5:
                        from ..models.bglike_structured_v5 import BGLikeStructuredV5

                        ability_emb_dim = int(kwargs.pop("ability_emb_dim", 8))
                        ability_attention_init_scale = float(
                            kwargs.pop("ability_attention_init_scale", 0.1)
                        )
                        net = BGLikeStructuredV5(
                            ability_emb_dim=ability_emb_dim,
                            ability_attention_init_scale=ability_attention_init_scale,
                            **v3_v4_kwargs,
                        )
                    elif is_bglike_structured_v6:
                        from ..models.bglike_structured_v6 import BGLikeStructuredV6

                        ability_emb_dim = int(kwargs.pop("ability_emb_dim", 8))
                        net = BGLikeStructuredV6(
                            ability_emb_dim=ability_emb_dim,
                            **v3_v4_kwargs,
                        )
                    elif is_bglike_structured_v7:
                        from ..models.bglike_structured_v7 import BGLikeStructuredV7
                        from ..models.bglike_structured_v8 import BGLikeStructuredV8
                        from ..models.bglike_structured_v9 import BGLikeStructuredV9

                        ability_emb_dim = int(kwargs.pop("ability_emb_dim", 8))
                        # Agent-level DvD knobs (not net constructor args except
                        # num_identities, which also sets the net's obs width).
                        dvd_num_identities = int(kwargs.pop("num_identities", 8))
                        dvd_diversity_coef = float(kwargs.pop("diversity_coef", 0.0))
                        dvd_diversity_ema = float(kwargs.pop("diversity_ema", 0.1))
                        dvd_identity_seed = int(kwargs.pop("identity_seed", 0))
                        dvd_identity_tribes = kwargs.pop("identity_tribes", None)
                        dvd_identity_init_std = float(kwargs.pop("identity_init_std", 0.0))
                        dvd_reward_mode = str(kwargs.pop("diversity_reward_mode", "final"))
                        if is_bglike_structured_v9:
                            net_cls = BGLikeStructuredV9
                        elif is_bglike_structured_v8:
                            net_cls = BGLikeStructuredV8
                        else:
                            net_cls = BGLikeStructuredV7
                        net = net_cls(
                            ability_emb_dim=ability_emb_dim,
                            num_identities=dvd_num_identities,
                            **v3_v4_kwargs,
                        )
                    else:
                        from ..models.bglike_structured_v3 import BGLikeStructuredV3

                        net = BGLikeStructuredV3(**v3_v4_kwargs)
                else:
                    from ..models.bglike_structured_v2 import BGLikeStructuredV2

                    net = BGLikeStructuredV2(**common_kwargs)
            else:
                slot_hidden_channels = int(kwargs.pop("slot_hidden_channels", 32))
                trunk_hidden_size = int(kwargs.pop("trunk_hidden_size", 256))
                state_dim = int(kwargs.pop("state_dim", 128))
                entity_attention_layers = int(kwargs.pop("entity_attention_layers", 0))
                use_global_entity_token = bool(kwargs.pop("use_global_entity_token", True))
                obs_layout = "bglike" if is_bglike_structured else "minibg"
                net = MiniBGStructuredActorCritic(
                    slot_hidden=slot_hidden_channels,
                    trunk_hidden=trunk_hidden_size,
                    region_conv2_kernel=region_conv2_kernel,
                    state_dim=state_dim,
                    action_dim=action_dim,
                    interaction_dim=interaction_dim,
                    order_hidden=order_hidden,
                    order_pos_dim=order_pos_dim,
                    score_hidden=score_hidden,
                    order_score_hidden=order_score_hidden,
                    critic_hidden=critic_hidden,
                    card_emb_dim=card_emb_dim,
                    entity_attention_layers=entity_attention_layers,
                    entity_attention_heads=entity_attention_heads,
                    entity_attention_ff_mult=entity_attention_ff_mult,
                    entity_attention_init_scale=entity_attention_init_scale,
                    use_global_entity_token=use_global_entity_token,
                    obs_layout=obs_layout,
                    num_pool_indices=num_pool_indices,
                )

            kwargs["observation_type"] = obs_type or "vector"
            ppo_kw = dict(net.get_constructor_kwargs())
            kwargs["network"] = net
            kwargs["ppo_network_type"] = ppo_network_type_for_save(network_type)
            kwargs["ppo_network_kwargs"] = ppo_kw
            if patch_build is not None:
                kwargs["patch_build"] = patch_build
            kwargs.pop("action_space", None)
            if is_bglike_structured_v7:
                return PPODvDAgent(
                    num_identities=dvd_num_identities,
                    diversity_coef=dvd_diversity_coef,
                    diversity_ema=dvd_diversity_ema,
                    identity_seed=dvd_identity_seed,
                    identity_tribes=dvd_identity_tribes,
                    identity_init_std=dvd_identity_init_std,
                    diversity_reward_mode=dvd_reward_mode,
                    **_filter_ppo_agent_kwargs(kwargs),
                )
            return MiniBGPPOStructuredAgent(**_filter_ppo_agent_kwargs(kwargs))

        if is_minibg:
            if obs_shape is None or num_actions is None:
                raise ValueError(
                    "PPO network_type minibg_slot requires num_actions and "
                    "a 1-D observation vector (inferred at train startup)."
                )
            slot_hidden_channels = int(kwargs.pop("slot_hidden_channels", 32))
            trunk_hidden_size = int(kwargs.pop("trunk_hidden_size", 256))
            region_conv2_kernel = int(kwargs.pop("region_conv2_kernel", 1))
            card_emb_dim = int(kwargs.pop("card_emb_dim", 16))
            net = build_ppo_actor_critic(
                PPO_NETWORK_MINIBG_SLOT,
                tuple(obs_shape),
                int(num_actions),
                slot_hidden_channels=slot_hidden_channels,
                trunk_hidden_size=trunk_hidden_size,
                region_conv2_kernel=region_conv2_kernel,
                card_emb_dim=card_emb_dim,
                num_pool_indices=num_pool_indices,
            )
            kwargs["observation_type"] = obs_type or "vector"
            ppo_kw = {
                "slot_hidden": slot_hidden_channels,
                "trunk_hidden": trunk_hidden_size,
                "region_conv2_kernel": region_conv2_kernel,
                "card_emb_dim": card_emb_dim,
                "num_pool_indices": net.num_pool_indices,
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
        if patch_build is not None:
            kwargs["patch_build"] = patch_build
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
    "MiniBGPPOStructuredAgent",
    "AlphaZeroAgent",
]
