from src.envs.minibg.replay_render import decode_env_action, render_jsonl_records

_PL = (
    '"hp":15,"gold":3,"tier":1,"shop_done":false,"shop_acts":0,'
    '"board":[{"card_id":"a","atk":1,"hp":1,"tier":1,"kw":[],"shield":false,"token":false,"abilities":[]}],'
    '"shop":[]'
)


def test_decode_env_action():
    assert decode_env_action(0) == "ROLL"
    assert decode_env_action(1) == "LEVEL_UP"
    assert decode_env_action(2) == "BUY_SHOP_0"
    assert decode_env_action(4) == "BUY_SHOP_2"
    assert decode_env_action(5) == "SELL_BOARD_0"
    assert decode_env_action(9).startswith("END_SHOP perm#0")


def test_render_battle_block_on_round_increment():
    s0 = (
        '{"type":"frame","p":0,"a":2,"illegal":false,'
        '"state":{"round":1,"cur":0,"p0":{' + _PL + '},"p1":{' + _PL + "}}}"
    )
    s1 = (
        '{"type":"frame","p":1,"a":9,"illegal":false,'
        '"state":{"round":2,"cur":0,"p0":{' + _PL + '},'
        '"p1":{"hp":14,"gold":4,"tier":1,"shop_done":false,"shop_acts":0,"board":[],"shop":[]}}}'
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
