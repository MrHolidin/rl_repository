from src.envs.minibg.replay_render import decode_env_action, render_jsonl_records

_PL = (
    '"hp":30,"gold":3,"tier":1,"shop_done":false,"shop_acts":0,'
    '"board":[{"card_id":"a","atk":1,"hp":1,"tier":1,"kw":[],"shield":false,"token":false,"abilities":[]}],'
    '"shop":[]'
)


def test_decode_env_action():
    from src.envs.minibg.action_map import (
        A_DISCOVER_BASE,
        A_FINISH,
        A_PLACE_BASE,
        A_SELECT_ORDER_BASE,
    )

    assert decode_env_action(0) == "ROLL"
    assert decode_env_action(1) == "LEVEL_UP"
    assert decode_env_action(2) == "BUY_SHOP_0"
    assert decode_env_action(4) == "BUY_SHOP_2"
    assert decode_env_action(5) == "SELL_BOARD_0"
    assert decode_env_action(A_PLACE_BASE) == "PLACE_HAND_0"
    assert decode_env_action(A_DISCOVER_BASE) == "DISCOVER_PICK_0"
    assert decode_env_action(A_FINISH) == "FINISH"
    assert decode_env_action(A_SELECT_ORDER_BASE).startswith("SELECT_ORDER perm#0")


def test_render_hand_finish_then_select_order_deduped():
    pl = (
        '"hp":30,"gold":0,"tier":1,"phase":"ORDER","shop_done":false,"shop_acts":3,'
        '"board":[],"shop":[],"hand":[{"card_id":"x","atk":2,"hp":3,"tier":1,"kw":[],"shield":false,"token":false,"abilities":[]},null,null]'
    )
    fin = (
        '{"type":"frame","p":0,"a":27,"illegal":false,'
        f'"state":{{"round":1,"cur":0,"p0":{{{pl}}},"p1":{{{_PL}}}}}}}'
    )
    sel = (
        '{"type":"frame","p":0,"a":28,"illegal":false,'
        f'"state":{{"round":1,"cur":0,"p0":{{{pl}}},"p1":{{{_PL}}}}}}}'
    )
    text = render_jsonl_records(
        ['{"type":"header","game":"minibg"}', fin, sel]
    )
    assert text.count("рука (на конец хода)") == 1
    assert "[x 2/3]" in text


def test_render_hand_on_finish():
    pl = (
        '"hp":30,"gold":0,"tier":1,"shop_done":false,"shop_acts":2,'
        '"board":[],"shop":[],"hand":[{"card_id":"x","atk":2,"hp":3,"tier":1,"kw":[],"shield":false,"token":false,"abilities":[]},null,null]'
    )
    frame = (
        '{"type":"frame","p":0,"a":27,"illegal":false,'
        f'"state":{{"round":1,"cur":0,"p0":{{{pl}}},"p1":{{{_PL}}}}}}}'
    )
    text = render_jsonl_records(['{"type":"header","game":"minibg"}', frame])
    assert "рука (на конец хода)" in text
    assert "[x 2/3]" in text


def test_render_battle_block_on_round_increment():
    s0 = (
        '{"type":"frame","p":0,"a":2,"illegal":false,'
        '"state":{"round":1,"cur":0,"p0":{' + _PL + '},"p1":{' + _PL + "}}}"
    )
    s1 = (
        '{"type":"frame","p":1,"a":9,"illegal":false,'
        '"state":{"round":2,"cur":0,"p0":{' + _PL + '},'
        '"p1":{"hp":29,"gold":4,"tier":1,"shop_done":false,"shop_acts":0,"board":[],"shop":[]}}}'
    )
    text = render_jsonl_records(
        [
            '{"type":"header","game":"minibg"}',
            s0,
            s1,
        ]
    )
    assert "Перед боем" in text
    assert "Бой: раунд 1→2" in text
    assert "P1 -1" in text
