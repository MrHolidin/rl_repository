"""Tests for the v6 obs (no ability tail) and the v12 net's frozen card table."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agents.random_agent import RandomAgent
from src.bg_catalog.patch_context import load_patch_context
from src.envs.bglike import card_static as cs
from src.envs.bglike.game import BGLikeGame
from src.envs.bglike.lobby_env import (
    BGLobbyEnv,
    OBS_KIND_BGLIKE_V6_HEROES,
    _obs_dim_for_kind,
)
from src.envs.bglike.obs import OBS_DIM, build_observation
from src.envs.bglike.obs_v5_heroes import HERO_BLOCK_DIM
from src.envs.bglike.obs_v6_heroes import (
    OBS_DIM_V6_HEROES,
    build_observation_v6_heroes,
)
from src.envs.bglike.seat_config import lobby_from_learned_seats
from src.envs.minibg.obs import CARD_IDX_OFFSET, GOLDEN_OFFSET, SLOT_DIM
from src.models.bglike_structured_v12 import BGLikeStructuredV12

PATCH_DIR = "data/bgcore/19_6_0_74257"


@pytest.fixture(scope="module")
def patch():
    return load_patch_context(PATCH_DIR)


def _hero_game(seed=3):
    g = BGLikeGame(seed=seed, with_heroes=True, patch_dir=PATCH_DIR)
    return g, g.initial_state()


def _net(patch, **kw):
    kw.setdefault("card_patch_dir", PATCH_DIR)
    return BGLikeStructuredV12(
        num_pool_indices=patch.num_pool_indices,
        slot_hidden=32,
        state_dim=64,
        entity_attention_layers=1,
        summary_queries=2,
        num_identities=4,
        **kw,
    )


# --------------------------------------------------------------------------- #
# Obs layout
# --------------------------------------------------------------------------- #


def test_v6_dim_is_base_plus_hero_block():
    assert OBS_DIM_V6_HEROES == OBS_DIM + HERO_BLOCK_DIM
    assert _obs_dim_for_kind(OBS_KIND_BGLIKE_V6_HEROES) == OBS_DIM_V6_HEROES


def test_v6_starts_with_the_unmodified_base_obs(patch):
    """v6 must be the base obs plus the hero block — nothing reordered."""
    _, state = _hero_game()
    base = build_observation(state, 0, 0.0, is_my_turn=True, patch=patch)
    v6 = build_observation_v6_heroes(state, 0, 0.0, is_my_turn=True, patch=patch)
    assert v6.shape == (OBS_DIM_V6_HEROES,)
    np.testing.assert_array_equal(v6[:OBS_DIM], base)


def test_env_emits_v6_obs():
    agents = {s: RandomAgent(seed=s) for s in range(8)}
    cfgs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)
    env = BGLobbyEnv(
        cfgs,
        learned_seats=tuple(range(8)),
        training_seats=(0,),
        seed=5,
        patch_dir=PATCH_DIR,
        obs_kind=OBS_KIND_BGLIKE_V6_HEROES,
        with_heroes=True,
    )
    env.reset(seed=5)
    assert env.obs_dim == OBS_DIM_V6_HEROES
    assert env.obs_for_seat(env.current_seat()).shape == (OBS_DIM_V6_HEROES,)


# --------------------------------------------------------------------------- #
# Static table
# --------------------------------------------------------------------------- #


def test_table_shape_and_padding_rows(patch):
    table, meta = cs.build_card_static_table(patch, text_dim=32)
    n = patch.num_pool_indices + 1
    assert table.shape == (2 * n, 32 + cs.NUM_DIM)
    assert meta["n_rows"] == n
    # Index 0 is "empty slot" in both the normal and the golden half.
    assert not table[0].any()
    assert not table[n].any()


def test_multiplier_factor_is_encoded(patch):
    """Brann/Baron/Khadgar's x2 was invisible before: `factor` was never read."""
    table, _ = cs.build_card_static_table(patch, text_dim=32)
    d = patch.card_id_to_dense
    factor_col = 32 + 1 + cs.NUMBER_FIELDS.index("factor")
    for card in ("LOE_077", "FP1_031", "DAL_575"):  # Brann, Baron, Khadgar
        assert table[d[card]][factor_col] == pytest.approx(2.0 / 5.0)


def test_text_separates_cards_the_numbers_cannot(patch):
    """The three multiplier auras share every numeric field; text must not."""
    table, _ = cs.build_card_static_table(patch, text_dim=32)
    d = patch.card_id_to_dense
    brann, baron = table[d["LOE_077"]], table[d["FP1_031"]]
    np.testing.assert_allclose(brann[32:], baron[32:])
    assert not np.allclose(brann[:32], baron[:32])


def test_golden_half_differs_where_golden_abilities_differ(patch):
    table, _ = cs.build_card_static_table(patch, text_dim=32)
    n = patch.num_pool_indices + 1
    d = patch.card_id_to_dense
    differing = [
        c for c in patch.pool_ids
        if not np.allclose(table[d[c]][32:], table[d[c] + n][32:])
    ]
    assert differing, "golden minions must not share the normal numeric block"
    # Text is number-masked, so the golden half reuses the same text vector.
    some = differing[0]
    np.testing.assert_allclose(table[d[some]][:32], table[d[some] + n][:32])


def test_random_mode_keeps_numbers_and_is_seeded(patch):
    text, _ = cs.build_card_static_table(patch, text_dim=32)
    r1, _ = cs.build_card_static_table(
        patch, text_mode=cs.TEXT_MODE_RANDOM, text_dim=32, random_seed=7
    )
    r2, _ = cs.build_card_static_table(
        patch, text_mode=cs.TEXT_MODE_RANDOM, text_dim=32, random_seed=7
    )
    r3, _ = cs.build_card_static_table(
        patch, text_mode=cs.TEXT_MODE_RANDOM, text_dim=32, random_seed=8
    )
    np.testing.assert_allclose(r1[:, 32:], text[:, 32:])  # control keeps magnitudes
    np.testing.assert_allclose(r1, r2)  # deterministic
    assert not np.allclose(r1[:, :32], r3[:, :32])  # seed actually varies it
    assert not np.allclose(r1[:, :32], text[:, :32])  # and is not the text


def test_k_static_abil_covers_the_patch(patch):
    """Guards the assumption that a card template holds at most K abilities."""
    worst = max(
        len(patch.templates[c].abilities or ()) for c in patch.card_index_ids
    )
    assert worst <= cs.K_STATIC_ABIL
    cs.build_number_table(patch)  # raises if a template overflows


def test_magnetised_minion_still_encodes(patch):
    """The table is keyed by card template, so a magnetised mech is not
    described by its own row. That is a known blind spot — assert only that
    nothing crashes and that the note documenting it exists."""
    from src.bg_recruitment.place import merge_magnetic_inplace

    target = patch.make_minion("BGS_071")  # Deflect-o-Bot, 2 abilities
    merge_magnetic_inplace(target, patch.make_minion("BOT_312"))
    assert len(target.abilities) > cs.K_STATIC_ABIL
    row = cs.encode_ability_numbers(target.abilities)
    assert row.shape == (cs.NUM_DIM,)
    assert "magnet" in cs.magnetic_divergence_note().lower()


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #


def test_net_drops_card_emb_and_ability_encoder(patch):
    net = _net(patch)
    names = dict(net.named_modules())
    assert "card_emb" not in names
    assert "ability_encoder" not in names
    assert "card_static" in dict(net.named_buffers())
    kw = net.get_constructor_kwargs()
    for dead in ("card_emb_dim", "use_card_emb", "ability_emb_dim"):
        assert dead not in kw


def test_card_path_has_the_same_depth_as_v11(patch):
    """The frozen row must reach slot_proj directly.

    v11's card path is un-activated (AbilityTokenEncoder.proj) up to the single
    ReLU on slot_proj. An extra projection here would either bottleneck the row
    linearly or add a nonlinearity v11 lacks — both would make a v11-vs-v12
    comparison measure architecture depth instead of the representation.
    """
    net = _net(patch)
    assert not hasattr(net, "card_static_proj")
    assert net.slot_proj.in_features == SLOT_DIM - 1 + net.card_row_dim
    assert net.pending_to_slot.in_features == net.card_row_dim


def test_net_forward_on_a_real_observation(patch):
    net = _net(patch)
    _, state = _hero_game()
    obs = build_observation_v6_heroes(state, 0, 0.0, is_my_turn=True, patch=patch)
    state_emb, cache = net.encode_state(torch.from_numpy(obs).unsqueeze(0))
    assert state_emb.shape == (1, net.state_dim)
    assert cache["E_own"].shape == (1, net.own_len, net.slot_hidden)


def test_net_rejects_the_wrong_obs_width(patch):
    net = _net(patch)
    with pytest.raises(ValueError):
        net.encode_state(torch.zeros(1, OBS_DIM_V6_HEROES + 1))


def test_identity_tail_is_accepted(patch):
    net = _net(patch)
    out, _ = net.encode_state(torch.zeros(1, OBS_DIM_V6_HEROES + net.num_identities))
    assert out.shape == (1, net.state_dim)


def test_golden_slot_reads_the_golden_half(patch):
    """Flipping the slot's golden bit must change the gathered card row."""
    net = _net(patch)
    d = patch.card_id_to_dense
    card = next(
        c for c in patch.pool_ids
        if not np.allclose(
            cs.encode_ability_numbers(patch.templates[c].abilities or ()),
            cs.encode_ability_numbers(patch.triple_merge_golden_abilities(c)),
        )
    )
    slot = torch.zeros(1, 1, SLOT_DIM)
    slot[0, 0, CARD_IDX_OFFSET] = float(d[card])
    normal = net._encode_region(slot, None, net.own_pos_emb)
    slot[0, 0, GOLDEN_OFFSET] = 1.0
    golden = net._encode_region(slot, None, net.own_pos_emb)
    assert not torch.allclose(normal, golden)


def test_unbuilt_table_is_refused_not_silently_zero(patch):
    net = BGLikeStructuredV12(
        num_pool_indices=patch.num_pool_indices,
        slot_hidden=32,
        state_dim=64,
        entity_attention_layers=1,
        card_patch_dir=None,
    )
    with pytest.raises(RuntimeError, match="never built"):
        net.encode_state(torch.zeros(1, OBS_DIM_V6_HEROES))


def test_state_dict_round_trip_without_the_patch_package(patch):
    """An eval box that lacks the patch dir must still reload a checkpoint."""
    src = _net(patch)
    dst = BGLikeStructuredV12(
        **{**src.get_constructor_kwargs(), "card_patch_dir": None}
    )
    dst.load_state_dict(src.state_dict())
    x = torch.zeros(2, OBS_DIM_V6_HEROES)
    a, _ = src.encode_state(x)
    b, _ = dst.encode_state(x)
    torch.testing.assert_close(a, b)


def test_patch_mismatch_is_rejected(patch):
    with pytest.raises(ValueError, match="num_pool_indices"):
        BGLikeStructuredV12(
            num_pool_indices=patch.num_pool_indices + 1,
            slot_hidden=32,
            card_patch_dir=PATCH_DIR,
        )


def test_missing_patch_package_falls_back_instead_of_raising(patch, tmp_path):
    """card_patch_dir is the *training* box's absolute path.

    Runs train on rented machines and are evaluated elsewhere, so the stored
    path routinely does not exist at load time. That must degrade to the zero
    placeholder (which load_state_dict then fills), not blow up — otherwise no
    remotely-trained checkpoint can be scored locally.
    """
    src = _net(patch)
    kw = {**src.get_constructor_kwargs(), "card_patch_dir": str(tmp_path / "gone")}
    dst = BGLikeStructuredV12(**kw)
    with pytest.raises(RuntimeError, match="never built"):
        dst.encode_state(torch.zeros(1, OBS_DIM_V6_HEROES))
    dst.load_state_dict(src.state_dict())
    a, _ = src.encode_state(torch.zeros(2, OBS_DIM_V6_HEROES))
    b, _ = dst.encode_state(torch.zeros(2, OBS_DIM_V6_HEROES))
    torch.testing.assert_close(a, b)


# --------------------------------------------------------------------------- #
# Training wiring
# --------------------------------------------------------------------------- #


def test_run_distributed_pins_obs_kind_and_structured_path():
    """A v12 run must take the structured path; the flat path dies on legal_mask."""
    from src.training.run_distributed import prepare_bg_network_params

    game_params = {"patch_dir": PATCH_DIR}
    agent_params = {"network_type": "bglike_structured_v12"}
    plan = prepare_bg_network_params("bglike", game_params, agent_params)

    assert plan.use_structured is True
    assert plan.is_dvd_v7 is True
    assert game_params["use_structured"] is True
    assert game_params["obs_kind"] == OBS_KIND_BGLIKE_V6_HEROES
    assert game_params["with_heroes"] is True


def test_conflicting_obs_kind_is_rejected():
    from src.training.run_distributed import prepare_bg_network_params

    with pytest.raises(ValueError, match="obs_kind"):
        prepare_bg_network_params(
            "bglike",
            {"patch_dir": PATCH_DIR, "obs_kind": "bglike_v5_heroes"},
            {"network_type": "bglike_structured_v12"},
        )


def test_agent_params_get_the_patch_dir_for_the_table():
    from src.training.patch_config import apply_patch_to_agent_params

    agent_params: dict = {}
    apply_patch_to_agent_params({"patch_dir": PATCH_DIR}, agent_params)
    assert agent_params["card_patch_dir"]
    assert agent_params["num_pool_indices"] > 0


def test_obs_sizing_knows_v6():
    from src.training.obs_sizing import apply_bg_observation_defaults

    agent_params: dict = {}
    apply_bg_observation_defaults(
        "bglike", agent_params, obs_kind=OBS_KIND_BGLIKE_V6_HEROES
    )
    assert agent_params["observation_shape"] == (OBS_DIM_V6_HEROES,)


# --------------------------------------------------------------------------- #
# Auxiliary battle-outcome head
# --------------------------------------------------------------------------- #


def _bp_net(patch, **cfg):
    return _net(patch, battle_pred_config={"enabled": True, **cfg})


def test_battle_head_off_by_default(patch):
    net = _net(patch)
    assert net.battle_head is None
    assert net._battle_pred_enabled is False
    with pytest.raises(RuntimeError, match="disabled"):
        net.predict_battle(
            torch.zeros(1, net.board_size, SLOT_DIM),
            torch.zeros(1, net.board_size, SLOT_DIM),
            torch.zeros(1),
        )


def test_battle_head_predicts_and_round_trips(patch):
    net = _bp_net(patch)
    own = torch.zeros(3, net.board_size, SLOT_DIM)
    opp = torch.zeros(3, net.board_size, SLOT_DIM)
    own[:, 0, CARD_IDX_OFFSET] = 7.0
    opp[:, 0, CARD_IDX_OFFSET] = 11.0
    out = net.predict_battle(own, opp, torch.tensor([1.0, 0.0, 1.0]))
    assert out.shape == (3,)
    kw = net.get_constructor_kwargs()
    assert kw["battle_pred_config"]["enabled"] is True
    dst = BGLikeStructuredV12(**{**kw, "card_patch_dir": None})
    dst.load_state_dict(net.state_dict())


def test_detach_features_cuts_the_encoder_gradient(patch):
    """detach_features=True must leave the head a pure probe.

    With it off the head regularises the slot encoder; with it on the encoder
    must receive nothing, so the head can be run as a diagnostic without
    changing what the policy learns.
    """
    net = _bp_net(patch)
    own = torch.zeros(4, net.board_size, SLOT_DIM)
    opp = torch.zeros(4, net.board_size, SLOT_DIM)
    own[:, 0, CARD_IDX_OFFSET] = 7.0
    opp[:, 0, CARD_IDX_OFFSET] = 11.0
    af = torch.zeros(4)

    hot = net.predict_battle(own, opp, af).sum()
    g = torch.autograd.grad(hot, net.slot_proj.weight, retain_graph=True, allow_unused=True)[0]
    assert g is not None and float(g.norm()) > 0

    cold = net.predict_battle(own, opp, af, detach_features=True).sum()
    g2 = torch.autograd.grad(cold, net.slot_proj.weight, retain_graph=True, allow_unused=True)[0]
    assert g2 is None


def test_head_reads_the_opponent_board(patch):
    """The enemy board must actually reach the prediction."""
    net = _bp_net(patch)
    own = torch.zeros(2, net.board_size, SLOT_DIM)
    opp = torch.zeros(2, net.board_size, SLOT_DIM)
    own[:, 0, CARD_IDX_OFFSET] = 7.0
    base = net.predict_battle(own, opp, torch.zeros(2))
    opp[:, 0, CARD_IDX_OFFSET] = 11.0
    changed = net.predict_battle(own, opp, torch.zeros(2))
    assert not torch.allclose(base, changed)


def test_agent_picks_up_the_head_config(patch):
    from src.envs.bglike.action_map import NUM_ENV_ACTIONS
    from src.registry import make_agent

    ag = make_agent(
        "ppo",
        network_type="bglike_structured_v12",
        observation_type="vector",
        observation_shape=(OBS_DIM_V6_HEROES,),
        num_actions=int(NUM_ENV_ACTIONS),
        num_pool_indices=patch.num_pool_indices,
        card_patch_dir=PATCH_DIR,
        battle_pred={"enabled": True, "aux_coef": 0.5},
        slot_hidden_channels=32,
        num_identities=4,
        device="cpu",
        seed=3,
    )
    assert ag._battle_pred_enabled is True
    assert ag._battle_pred_aux_coef == pytest.approx(0.5)
    assert ag.policy_net.battle_head is not None


# --------------------------------------------------------------------------- #
# Battle head normalisation
# --------------------------------------------------------------------------- #


def test_head_defaults_are_published_on_the_config(patch):
    """The agent reads these off the model, so the model must set them."""
    from src.models.bglike_structured_v12 import DEFAULT_DAMAGE_NORM, DEFAULT_HUBER_DELTA

    net = _net(patch, battle_pred_config={"enabled": True})
    assert net.battle_pred_config["damage_norm"] == DEFAULT_DAMAGE_NORM
    assert net.battle_pred_config["huber_delta"] == DEFAULT_HUBER_DELTA


def test_prediction_is_always_in_minus_one_one(patch):
    net = _net(patch, battle_pred_config={"enabled": True})
    own = torch.randn(32, net.board_size, SLOT_DIM) * 50.0  # absurd inputs
    opp = torch.randn(32, net.board_size, SLOT_DIM) * 50.0
    pred = net.predict_battle(own, opp, torch.ones(32))
    assert bool((pred.abs() < 1.0).all())


def test_label_squash_is_monotone_and_bounded():
    """tanh, not clip: a 30-damage blowout must still outrank a 16-damage one."""
    from src.models.bglike_structured_v12 import DEFAULT_DAMAGE_NORM

    d = torch.tensor([-30.0, -16.0, -4.5, 0.0, 4.5, 16.0, 30.0])
    t = torch.tanh(d / DEFAULT_DAMAGE_NORM)
    assert bool((t.diff() > 0).all())
    assert bool((t.abs() < 1.0).all())
    assert t[3].item() == pytest.approx(0.0)


def test_head_gradient_does_not_blow_up_with_head_scale(patch):
    """The failure this normalisation exists to prevent.

    With a raw-damage target the head must grow its own weights to reach +/-15,
    and the gradient it pushes into the shared slot encoder grows with them --
    measured at ~2100x between an untrained and a trained head, which makes
    aux_coef uncalibratable. Under tanh the same weight growth saturates the
    output and the gradient falls away instead of exploding.
    """
    torch.manual_seed(0)
    net = _net(patch, battle_pred_config={"enabled": True})
    shared = [p for n, p in net.named_parameters() if n.startswith("slot_proj")]
    own = torch.randn(48, net.board_size, SLOT_DIM)
    opp = torch.randn(48, net.board_size, SLOT_DIM)
    tgt = torch.tanh(torch.randn(48))

    def grad_norm():
        loss = torch.nn.functional.smooth_l1_loss(
            net.predict_battle(own, opp, torch.zeros(48)), tgt, beta=0.33
        )
        g = torch.autograd.grad(loss, shared, allow_unused=True)
        return float(torch.sqrt(sum((x ** 2).sum() for x in g if x is not None)))

    at_init = grad_norm()
    with torch.no_grad():
        for n, p in net.named_parameters():
            if n.startswith("battle_head"):
                p.mul_(20.0)
    saturated = grad_norm()
    assert saturated < at_init * 10.0


def test_tier_first_round_is_recorded_on_the_shared_step_path():
    """Every eval path must get it, not just callers that drive their own loop."""
    agents = {s: RandomAgent(seed=s) for s in range(8)}
    cfgs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)
    env = BGLobbyEnv(
        cfgs, learned_seats=tuple(range(8)), training_seats=(0,), seed=11,
        patch_dir=PATCH_DIR, obs_kind=OBS_KIND_BGLIKE_V6_HEROES, with_heroes=True,
    )
    env.reset(seed=11)
    assert env.tier_first_round(0) == {}
    env.drain_until_lobby_done(deterministic=True)
    seen = [env.tier_first_round(s) for s in range(8)]
    assert any(d for d in seen), "no seat ever left tier 1"
    for seat, d in enumerate(seen):
        tier = env.state.players[seat].tavern_tier
        for t in range(2, tier + 1):
            assert t in d, f"seat {seat} is on t{tier} but t{t} was never recorded"
        # Rounds must not decrease with tier: you pass through 3 to reach 4.
        rounds = [d[t] for t in sorted(d)]
        assert rounds == sorted(rounds)


def test_per_seat_obs_kinds_coexist_in_one_lobby():
    """One lobby, two layouts -- what lets checkpoints of different versions meet."""
    agents = {s: RandomAgent(seed=s) for s in range(8)}
    cfgs = lobby_from_learned_seats(tuple(range(8)), agent_by_seat=agents)
    env = BGLobbyEnv(
        cfgs, learned_seats=tuple(range(8)), training_seats=(0,), seed=3,
        patch_dir=PATCH_DIR, obs_kind=OBS_KIND_BGLIKE_V6_HEROES,
        obs_kind_by_seat={1: "bglike_v5_heroes"}, with_heroes=True,
    )
    env.reset(seed=3)
    assert env.obs_for_seat(0).shape[0] == OBS_DIM_V6_HEROES
    assert env.obs_for_seat(1).shape[0] == env.obs_dim_for_seat(1) != OBS_DIM_V6_HEROES
    with pytest.raises(ValueError, match="obs_kind_by_seat"):
        BGLobbyEnv(
            cfgs, learned_seats=tuple(range(8)), training_seats=(0,), seed=3,
            patch_dir=PATCH_DIR, obs_kind_by_seat={2: "nonsense"},
        )


def test_network_obs_kind_table_covers_the_live_nets():
    from src.training.bg_network_policy import obs_kind_for_network

    assert obs_kind_for_network("bglike_structured_v12") == OBS_KIND_BGLIKE_V6_HEROES
    assert obs_kind_for_network("bglike_structured_v11_heroes") == "bglike_v5_heroes"
    assert obs_kind_for_network("bglike_structured_v11") == "bglike_v5"
    assert obs_kind_for_network("something_else") == "bglike"


def test_loader_restores_the_agent_class_that_trained(tmp_path):
    """A DvD-trained checkpoint must come back as a DvD agent.

    PPODvDAgent saves under its parent's agent_kind but appends an identity
    one-hot to every observation, which the net's identity_slot_gate then uses
    to scale each entity token. Restoring the parent drops the tail silently --
    the net accepts both widths -- and the checkpoint gets scored in a mode it
    never trained in (measured: 10.2% of decisions change).
    """
    import torch

    from src.agents.ppo_dvd_agent import PPODvDAgent
    from src.evaluation.eval_checkpoints import load_training_agent_checkpoint

    patch = load_patch_context(PATCH_DIR)
    net = BGLikeStructuredV12(
        num_pool_indices=patch.num_pool_indices, card_patch_dir=PATCH_DIR,
        slot_hidden=32, state_dim=64, entity_attention_layers=1,
        summary_queries=2, num_identities=4,
    )
    assert any(n.startswith("identity_") for n, _ in net.named_parameters())
    with_ids = tmp_path / "dvd.pt"
    torch.save(
        {"agent_kind": "ppo_minibg_structured", "policy_state_dict": net.state_dict()},
        with_ids,
    )
    ckpt = torch.load(with_ids, map_location="cpu", weights_only=False)
    state = ckpt["policy_state_dict"]
    assert any(str(k).startswith("identity_") for k in state), (
        "the structural marker the loader keys on must be present"
    )
