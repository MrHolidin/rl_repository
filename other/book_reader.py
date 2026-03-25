"""
Pure-Python reader for Pascal Pons' Connect4 opening book (.book format).

Book format (from OpeningBook.hpp):
  6-byte header:  width, height, depth, partial_key_bytes, value_bytes, log_size
  size * partial_key_bytes:  keys  (truncated)
  size * 1:                  values (uint8; 0 = not stored)

where size = smallest prime >= 2^log_size.

Score encoding:
  WIDTH, HEIGHT = 7, 6
  MIN_SCORE = -(WIDTH*HEIGHT)//2 + 3  = -18
  stored_value = score - MIN_SCORE + 1  (so 0 means absent)
  score = stored_value + MIN_SCORE - 1  = stored_value - 19

Position encoding (Pons bitmask):
  Column c, row r (0=bottom, HEIGHT-1=top): bit = 1 << (c*(HEIGHT+1) + r)
  current_position: current player's pieces
  mask:             all pieces

key3: symmetric base-3 encoding – minimum of left-to-right and right-to-left
      column scans, divided by 3 (last digit always 0).
"""

import struct

WIDTH  = 7
HEIGHT = 6
MIN_SCORE = -(WIDTH * HEIGHT) // 2 + 3   # -18
MAX_SCORE = (WIDTH * HEIGHT + 1) // 2 - 3  # 18


def _next_prime(n: int) -> int:
    def is_prime(x):
        if x < 2:
            return False
        i = 2
        while i * i <= x:
            if x % i == 0:
                return False
            i += 1
        return True
    while not is_prime(n):
        n += 1
    return n


def load_book(path: str):
    with open(path, "rb") as f:
        data = f.read()
    w, h, depth, pk_bytes, v_bytes, log_size = data[:6]
    assert w == WIDTH and h == HEIGHT and v_bytes == 1
    size = _next_prime(1 << log_size)
    offset = 6
    keys_raw = data[offset: offset + size * pk_bytes]
    vals_raw = data[offset + size * pk_bytes: offset + size * pk_bytes + size]
    return dict(
        depth=depth,
        pk_bytes=pk_bytes,
        size=size,
        keys=keys_raw,
        vals=vals_raw,
    )


def _partial_key3(key: int, col: int, current_position: int, mask: int) -> int:
    pos = 1 << (col * (HEIGHT + 1))
    while pos & mask:
        key *= 3
        if pos & current_position:
            key += 1
        else:
            key += 2
        pos <<= 1
    key *= 3
    return key


def compute_key3(current_position: int, mask: int) -> int:
    kf = 0
    for c in range(WIDTH):
        kf = _partial_key3(kf, c, current_position, mask)
    kr = 0
    for c in range(WIDTH - 1, -1, -1):
        kr = _partial_key3(kr, c, current_position, mask)
    return min(kf, kr) // 3


def book_get(book: dict, key3_val: int) -> int:
    """Return stored value (0 = not present)."""
    pk_bytes = book["pk_bytes"]
    size     = book["size"]
    keys     = book["keys"]
    vals     = book["vals"]
    pos = key3_val % size
    mask_bits = (1 << (pk_bytes * 8)) - 1
    stored = int.from_bytes(keys[pos * pk_bytes: (pos + 1) * pk_bytes], "little")
    if stored == (key3_val & mask_bits):
        return vals[pos]
    return 0


def decode_score(raw: int):
    """raw=0 → None (not in book). Otherwise → score from current player's POV."""
    if raw == 0:
        return None
    return raw + MIN_SCORE - 1


def seq_to_bitmask(seq: str):
    """Replay a Pons move sequence (1-indexed cols) → (current_position, mask, nb_moves)."""
    current_position = 0
    mask = 0
    for ch in seq:
        col = int(ch) - 1
        bottom = 1 << (col * (HEIGHT + 1))
        col_mask = ((1 << HEIGHT) - 1) << (col * (HEIGHT + 1))
        move_bit = (mask + bottom) & col_mask
        current_position ^= mask
        mask |= move_bit
    nb_moves = bin(mask).count("1")
    return current_position, mask, nb_moves


def kaggle_board_to_bitmask(board, mark, ROWS=6, COLS=7):
    """
    Convert Kaggle flat board (row 0 = top) to Pons bitmask.
    mark = current player's mark (1 or 2).
    Returns (current_position, mask, nb_moves).
    """
    opp = 3 - mark
    current_position = 0
    mask = 0
    for col in range(COLS):
        for row_from_bottom in range(ROWS):
            row_kaggle = ROWS - 1 - row_from_bottom
            cell = board[row_kaggle * COLS + col]
            if cell != 0:
                bit = 1 << (col * (HEIGHT + 1) + row_from_bottom)
                mask |= bit
                if cell == mark:
                    current_position |= bit
    nb_moves = bin(mask).count("1")
    return current_position, mask, nb_moves


def best_book_move(book: dict, current_position: int, mask: int, nb_moves: int):
    """
    Given the current position bitmask, enumerate legal moves and return the
    column (0-indexed) with the best book score, or None if not in book.

    We look up the RESULTING position after each move (opponent's turn).
    Their score from THEIR POV is raw_score; for us it's negated.
    """
    if nb_moves >= book["depth"]:
        return None, None

    best_col   = None
    best_score = None

    for col in range(WIDTH):
        # check column not full
        top_bit = 1 << (col * (HEIGHT + 1) + HEIGHT - 1)
        if mask & top_bit:
            continue  # full

        bottom = 1 << (col * (HEIGHT + 1))
        col_mask = ((1 << HEIGHT) - 1) << (col * (HEIGHT + 1))
        move_bit = (mask + bottom) & col_mask

        new_cp   = current_position ^ mask   # swap current / opponent
        new_mask = mask | move_bit

        raw = book_get(book, compute_key3(new_cp, new_mask))
        score = decode_score(raw)
        if score is None:
            continue

        # score is from opponent's POV → for us it's -score
        our_score = -score
        if best_score is None or our_score > best_score:
            best_score = our_score
            best_col   = col

    return best_col, best_score
