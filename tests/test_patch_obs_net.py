import pytest

from src.agents.ppo_structured_minibg_agent import MiniBGPPOStructuredAgent
from src.bg_catalog.patch_context import load_patch_context
from src.envs.bglike.game import BGLikeGame
from src.envs.bglike.obs import OBS_DIM, build_observation
from src.envs.minibg.obs import encode_minion
from src.models.bglike_structured_v2 import BGLikeStructuredV2
from src.training.patch_config import (
    apply_patch_to_agent_params,
    assert_checkpoint_patch_build,
)

_PATCH_36393 = "data/bgcore/15_6_2_36393"


def test_encode_minion_uses_explicit_card_index():
    ctx = load_patch_context(_PATCH_36393)
    m = ctx.make_minion("EX1_162")
    v = encode_minion(m, card_id_to_dense=ctx.card_id_to_dense)
    assert int(v[1]) == ctx.card_id_to_dense["EX1_162"]


def test_game_obs_uses_patch_card_index():
    game = BGLikeGame(seed=0, patch_dir=_PATCH_36393)
    ctx = load_patch_context(_PATCH_36393)
    state = game.initial_state()
    obs = build_observation(
        state,
        0,
        0.0,
        is_my_turn=True,
        patch=game._patch,
    )
    assert obs.shape[0] > 0
    assert game._patch.card_id_to_dense["EX1_162"] == ctx.card_id_to_dense["EX1_162"]


def test_structured_v2_card_emb_sized_from_patch():
    ctx = load_patch_context(_PATCH_36393)
    net = BGLikeStructuredV2(num_pool_indices=ctx.num_pool_indices)
    assert net.card_emb.num_embeddings == ctx.num_pool_indices + 1


def test_checkpoint_patch_build_mismatch_rejected(tmp_path):
    ctx = load_patch_context(_PATCH_36393)
    net = BGLikeStructuredV2(num_pool_indices=ctx.num_pool_indices)
    agent = MiniBGPPOStructuredAgent(
        observation_shape=(OBS_DIM,),
        observation_type="vector",
        num_actions=64,
        network=net,
        patch_build=ctx.build,
    )
    path = tmp_path / "agent.pt"
    agent.save(str(path))

    with pytest.raises(ValueError, match="patch_build"):
        MiniBGPPOStructuredAgent.load(
            str(path),
            device="cpu",
            patch_build=ctx.build + 1,
        )


def test_assert_checkpoint_patch_build_helper():
    assert_checkpoint_patch_build({"patch_build": 36393}, 36393)
    with pytest.raises(ValueError):
        assert_checkpoint_patch_build({"patch_build": 36393}, 74257)


def test_apply_patch_to_agent_params_fills_missing():
    ctx = load_patch_context(_PATCH_36393)
    game_params = {"patch_dir": str(ctx.patch_dir)}
    agent_params: dict = {}
    out = apply_patch_to_agent_params(game_params, agent_params)
    assert out.build == ctx.build
    assert agent_params["num_pool_indices"] == ctx.num_pool_indices
    assert agent_params["patch_build"] == ctx.build


def test_apply_patch_to_agent_params_accepts_matching_values():
    ctx = load_patch_context(_PATCH_36393)
    game_params = {"patch_dir": str(ctx.patch_dir)}
    agent_params = {
        "num_pool_indices": ctx.num_pool_indices,
        "patch_build": ctx.build,
    }
    apply_patch_to_agent_params(game_params, agent_params)
    assert agent_params["num_pool_indices"] == ctx.num_pool_indices
    assert agent_params["patch_build"] == ctx.build


def test_apply_patch_to_agent_params_rejects_num_pool_indices_mismatch():
    ctx = load_patch_context(_PATCH_36393)
    game_params = {"patch_dir": str(ctx.patch_dir)}
    agent_params = {"num_pool_indices": ctx.num_pool_indices + 1}
    with pytest.raises(ValueError, match="num_pool_indices"):
        apply_patch_to_agent_params(game_params, agent_params)


def test_apply_patch_to_agent_params_rejects_patch_build_mismatch():
    ctx = load_patch_context(_PATCH_36393)
    game_params = {"patch_dir": str(ctx.patch_dir)}
    agent_params = {"patch_build": ctx.build + 1}
    with pytest.raises(ValueError, match="patch_build"):
        apply_patch_to_agent_params(game_params, agent_params)


def test_build_observation_74257_game():
    game = BGLikeGame(seed=0, patch_dir="data/bgcore/19_6_0_74257")
    state = game.initial_state()
    obs = build_observation(state, 0, 0.0, is_my_turn=True, patch=game._patch)
    assert obs.shape == (OBS_DIM,)
    assert game._patch.build == 74257
    assert len(game._patch.meta.rotation_tribes) == 7


def test_shop_rotation_74257_excluded_dragon():
    from src.envs.bglike.obs import BGLIKE_GLOBAL_CORE_DIM
    from src.envs.minibg.obs import SHOP_ROTATION_OBS_DIM
    from src.bg_core.minion import Race

    game = BGLikeGame(
        seed=0,
        patch_dir="data/bgcore/19_6_0_74257",
        shop_excluded_race=Race.DRAGON,
    )
    state = game.initial_state()
    obs = build_observation(state, 0, 0.0, is_my_turn=True, patch=game._patch)
    obs_rot = obs[BGLIKE_GLOBAL_CORE_DIM : BGLIKE_GLOBAL_CORE_DIM + SHOP_ROTATION_OBS_DIM]
    idx = game._patch.meta.rotation_tribes.index(Race.DRAGON)
    assert idx == 4
    assert obs_rot[idx] == 1.0
    assert obs_rot[7] == pytest.approx(6 / 7)


def test_shop_rotation_globals_follow_patch_meta():
    from src.envs.minibg.obs import GLOBAL_CORE_DIM, _encode_shop_rotation_globals
    from src.envs.minibg.state import Race

    tribes = (Race.DEMON, Race.BEAST, Race.MURLOC, Race.MECHANICAL)
    v = _encode_shop_rotation_globals(
        Race.BEAST,
        rotation_tribes=tribes,
        cnt_active_shop_tribes=3,
    )
    rot = v
    assert rot[1] == 1.0
    assert rot[0] == 0.0
    assert rot[7] == pytest.approx(0.75)

    game = BGLikeGame(seed=0, patch_dir=_PATCH_36393, shop_excluded_race=Race.MURLOC)
    state = game.initial_state()
    obs = build_observation(state, 0, 0.0, is_my_turn=True, patch=game._patch)
    from src.envs.bglike.obs import BGLIKE_GLOBAL_CORE_DIM
    from src.envs.minibg.obs import SHOP_ROTATION_OBS_DIM

    obs_rot = obs[BGLIKE_GLOBAL_CORE_DIM : BGLIKE_GLOBAL_CORE_DIM + SHOP_ROTATION_OBS_DIM]
    idx = game._patch.meta.rotation_tribes.index(Race.MURLOC)
    assert obs_rot[idx] == 1.0
    assert obs_rot[7] == pytest.approx(
        game._patch.meta.cnt_active_shop_tribes
        / len(game._patch.meta.rotation_tribes)
    )
