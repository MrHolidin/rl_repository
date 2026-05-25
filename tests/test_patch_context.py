from src.bg_catalog.cards import build_card_templates
from pathlib import Path
from src.bg_catalog.patch_context import load_patch_context
from src.bg_catalog.patch_catalog import patch_build, patch_version

_PATCH_36393 = "data/bgcore/15_6_2_36393"
_PATCH_36393_PATH = Path(__file__).resolve().parents[1] / _PATCH_36393


def test_patch_dir_36393_layout():
    assert (_PATCH_36393_PATH / "catalog.json").is_file()
    assert (_PATCH_36393_PATH / "meta.json").is_file()
    assert (_PATCH_36393_PATH / "bindings.py").is_file()


def test_patch_context_matches_legacy_shims():
    ctx = load_patch_context(_PATCH_36393)
    assert patch_build() == ctx.build == 36393
    assert patch_version() == ctx.patch == "15.6.2"
    assert build_card_templates(patch=ctx) == dict(ctx.templates)


def test_patch_card_descriptions_include_names():
    ctx = load_patch_context(_PATCH_36393)
    assert len(ctx.descriptions) == len(ctx.templates)
    for card_id, desc in ctx.descriptions.items():
        assert desc.card_id == card_id
        assert desc.name.strip()
        assert desc.template.name == desc.name
        assert ctx.templates[card_id].name == desc.name


def test_nathrezim_overseer_description():
    ctx = load_patch_context(_PATCH_36393)
    desc = ctx.describe("BGS_001")
    assert desc.name == "Nathrezim Overseer"
    assert desc.in_tavern_pool is True
    assert desc.catalog_text is not None
    assert "Battlecry" in desc.catalog_text


def test_pool_ids_count():
    ctx = load_patch_context(_PATCH_36393)
    pool = [m for m in ctx.templates.values() if not m.is_token and not m.is_golden]
    assert len(ctx.pool_ids) == 81
    assert len(pool) >= 81


def test_card_index_dense_map():
    ctx = load_patch_context(_PATCH_36393)
    assert len(ctx.card_index_ids) == ctx.num_pool_indices == len(ctx.templates)
    assert ctx.card_id_to_dense["BGS_004"] >= 1
    assert "BGS_004" in ctx.card_index_ids


def test_meta_rotation_and_pool_copies():
    ctx = load_patch_context(_PATCH_36393)
    assert len(ctx.meta.rotation_tribes) == 4
    assert ctx.meta.rotation_excluded_count == 1
    assert ctx.meta.cnt_active_shop_tribes == 3
    assert ctx.meta.pool_copies_by_tier[1] == 16
    assert ctx.meta.pool_copies_by_tier[6] == 7


def test_game_records_patch_build():
    from src.envs.bglike.game import BGLikeGame

    game = BGLikeGame(seed=0, patch_dir=_PATCH_36393)
    state = game.initial_state()
    assert state.patch_build == 36393


def test_meta_drives_pool_copies(tmp_path):
    import json
    from pathlib import Path

    from src.bg_catalog.patch_context import PatchContext

    src = Path(__file__).resolve().parents[1] / "data" / "bgcore" / "15_6_2_36393"
    dst = tmp_path / "custom_patch"
    dst.mkdir()
    (dst / "catalog.json").write_text((src / "catalog.json").read_text(encoding="utf-8"))
    (dst / "bindings.py").write_text((src / "bindings.py").read_text(encoding="utf-8"))
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    meta["pool_copies_by_tier"]["1"] = 9
    (dst / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    ctx = PatchContext.load(dst)
    from src.bg_lobby.shared_pool import build_initial_shared_pool

    pool = build_initial_shared_pool(None, patch=ctx)
    assert pool.remaining_copies("EX1_162") == 9


def test_game_patch_dir_from_config():
    from src.envs.bglike.game import BGLikeGame

    game = BGLikeGame(seed=0, patch_dir="data/bgcore/15_6_2_36393")
    state = game.initial_state()
    assert state.patch_build == 36393
