from src.envs.minibg.replay_render import (
    decode_env_action,
    decode_env_action_compact,
    iter_pre_battle_rows,
    render_jsonl_records,
)

_PL = (
    '"hp":30,"gold":3,"tier":1,"shop_done":false,"shop_acts":0,'
    '"board":[{"card_id":"a","atk":1,"hp":1,"tier":1,"kw":[],"shield":false,"token":false,"abilities":[]}],'
    '"shop":[null,null,null,null,null,null]'
)


_HAND5 = (
    '[{"card_id":"x","atk":2,"hp":3,"tier":1,"kw":[],"shield":false,'
    '"token":false,"abilities":[]},null,null,null,null]'
)


def test_decode_env_action():
    from src.envs.minibg.action_map import A_DISCOVER_BASE, A_FINISH, A_PLACE_BASE, A_SWAP_BOARD_0
    from src.envs.minibg.actions import Action as GA

    assert decode_env_action(int(GA.ROLL)) == "ROLL"
    assert decode_env_action(int(GA.LEVEL_UP)) == "LEVEL_UP"
    assert decode_env_action(int(GA.BUY_SLOT_0)) == "BUY_SHOP_0"
    assert decode_env_action(int(GA.BUY_SLOT_2)) == "BUY_SHOP_2"
    assert decode_env_action(int(GA.SELL_BOARD_0)) == "SELL_BOARD_0"
    assert decode_env_action(int(A_PLACE_BASE)) == "PLACE_HAND_0"
    assert decode_env_action(int(A_DISCOVER_BASE)) == "DISCOVER_PICK_0"
    assert decode_env_action(int(A_FINISH)) == "FINISH"
    assert decode_env_action(int(GA.FINISH_FREEZE_SHOP)) == "FINISH_FREEZE_SHOP"
    assert decode_env_action(int(A_SWAP_BOARD_0)) == "SWAP_BOARD_0_1"


def test_decode_env_action_compact_no_slots():
    from src.envs.minibg.actions import Action as GA
    from src.envs.minibg.action_map import A_PLACE_BASE, A_SWAP_BOARD_0

    assert decode_env_action_compact(int(GA.BUY_SLOT_0)) == "BUY"
    assert decode_env_action_compact(int(GA.FINISH_FREEZE_SHOP)) == "FINISH_FREEZE_SHOP"
    assert decode_env_action_compact(int(A_PLACE_BASE)) == "PLACE"
    assert decode_env_action_compact(int(A_SWAP_BOARD_0)) == "SWAP_BOARD"


def test_render_hand_finish_then_order_submit_each_frame():
    from src.envs.minibg.actions import Action as GA

    pl = (
        '"hp":30,"gold":0,"tier":1,"phase":"SHOP","shop_done":false,"shop_acts":3,'
        f'"board":[],"shop":[null,null,null,null,null,null],"hand":{_HAND5}'
    )
    fin = (
        '{"type":"frame","p":0,"a":'
        f'{int(GA.FINISH)},"illegal":false,'
        f'"state":{{"round":1,"cur":0,"p0":{{{pl}}},"p1":{{{_PL}}}}}}}'
    )
    fin2 = (
        '{"type":"frame","p":0,"a":'
        f'{int(GA.FINISH)},"illegal":false,'
        f'"state":{{"round":1,"cur":0,"p0":{{{pl}}},"p1":{{{_PL}}}}}}}'
    )
    text = render_jsonl_records(
        ['{"type":"header","game":"minibg"}', fin, fin2]
    )
    assert text.count("рука P0:") == 2
    assert text.count("[x 2/3]") >= 2


def test_render_hand_on_finish():
    from src.envs.minibg.actions import Action as GA

    pl = (
        '"hp":30,"gold":0,"tier":1,"shop_done":false,"shop_acts":2,'
        f'"board":[],"shop":[null,null,null,null,null,null],"hand":{_HAND5}'
    )
    frame = (
        '{"type":"frame","p":0,"a":'
        f'{int(GA.FINISH)},"illegal":false,'
        f'"state":{{"round":1,"cur":0,"p0":{{{pl}}},"p1":{{{_PL}}}}}}}'
    )
    text = render_jsonl_records(['{"type":"header","game":"minibg"}', frame])
    assert "рука P0:" in text
    assert "[x 2/3]" in text


def test_render_battle_block_on_round_increment():
    s0 = (
        '{"type":"frame","p":0,"a":2,"illegal":false,'
        '"state":{"round":1,"cur":0,"p0":{' + _PL + '},"p1":{' + _PL + "}}}"
    )
    s1 = (
        '{"type":"frame","p":1,"a":9,"illegal":false,'
        '"state":{"round":2,"cur":0,"p0":{' + _PL + '},'
        '"p1":{"hp":29,"gold":4,"tier":1,"shop_done":false,"shop_acts":0,"board":[],"shop":[null,null,null,null,null,null]}}}'
    )
    text = render_jsonl_records(
        [
            '{"type":"header","game":"minibg"}',
            s0,
            s1,
        ]
    )
    assert "Перед боем" in text
    assert "в конце раунда набора 1" in text
    assert "Бой: раунд 1→2" in text
    assert "P1 -1" in text
    assert "лавка P0:" in text
    assert "лавка P1:" in text


def test_iter_pre_battle_rows_same_logic_as_render():
    s0 = (
        '{"type":"frame","p":0,"a":2,"illegal":false,'
        '"state":{"round":1,"cur":0,"p0":{' + _PL + '},"p1":{' + _PL + "}}}"
    )
    s1 = (
        '{"type":"frame","p":1,"a":9,"illegal":false,'
        '"state":{"round":2,"cur":0,"p0":{' + _PL + '},'
        '"p1":{"hp":29,"gold":4,"tier":1,"shop_done":false,"shop_acts":0,"board":[],"shop":[null,null,null,null,null,null]}}}'
    )
    lines = ['{"type":"header","game":"minibg"}', s0, s1]
    rows = list(iter_pre_battle_rows(iter(lines)))
    assert len(rows) == 1
    assert rows[0]["shop_round_ended"] == 1
    assert rows[0]["round_after"] == 2
    assert rows[0]["delta_hp_p1"] == -1
    assert "[a 1/1]" in rows[0]["p0_table"]
    assert "[a 1/1]" in rows[0]["p1_table"]
