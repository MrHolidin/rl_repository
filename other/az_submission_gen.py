"""
Generate a self-contained Kaggle ConnectX submission from an AlphaZero checkpoint.

Usage (in Kaggle notebook or locally):
    CHECKPOINT  = '/kaggle/input/.../alphazero_iter_000100.pt'
    MCTS_SIMS   = 400
    TEMPERATURE = 0.0   # greedy
    OUTPUT      = 'submission.py'
"""

import base64
import io
import json
import os
import torch

# ── settings ──────────────────────────────────────────────────────────────────
CHECKPOINT  = 'runs/alphazero/connect4/long_run/alphazero_iter_000100.pt'
BOOK_PATH   = '/tmp/7x6_small.book'   # set None to skip
MCTS_SIMS   = 400
TEMPERATURE = 0.0
C_PUCT      = 1.4
OUTPUT      = 'submission.py'
# ──────────────────────────────────────────────────────────────────────────────

ck = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
network_kwargs = ck['network_kwargs']

buf = io.BytesIO()
torch.save(ck['network_state_dict'], buf)
weights_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
kwargs_json = json.dumps(network_kwargs)

if BOOK_PATH and os.path.exists(BOOK_PATH):
    book_b64 = base64.b64encode(open(BOOK_PATH, 'rb').read()).decode('ascii')
    print(f"Opening book loaded: {os.path.basename(BOOK_PATH)}  ({len(book_b64)//1024} KB base64)")
else:
    book_b64 = ''
    print("No opening book.")

print(f"Checkpoint loaded: {os.path.basename(CHECKPOINT)}")
print(f"Network kwargs: {network_kwargs}")
print(f"Weights size: {len(weights_b64) // 1024} KB (base64)")

# ── submission template ───────────────────────────────────────────────────────
SUBMISSION = f'''"""Kaggle ConnectX — AlphaZero + MCTS agent (self-contained)."""

MCTS_SIMS   = {MCTS_SIMS}
TEMPERATURE = {TEMPERATURE}
C_PUCT      = {C_PUCT}
NETWORK_KWARGS = {kwargs_json}
WEIGHTS_B64 = "{weights_b64}"
BOOK_B64    = "{book_b64}"


def my_agent(observation, configuration):
    import math, base64, io, json
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    ROWS = configuration.rows
    COLS = configuration.columns

    # ── Network architecture ─────────────────────────────────────────────────

    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2   = nn.BatchNorm2d(channels)
        def forward(self, x):
            r = x
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            return F.relu(x + r)

    class AZNetwork(nn.Module):
        def __init__(self, rows, cols, in_channels, num_actions,
                     trunk_channels, num_res_blocks,
                     policy_channels, value_channels, value_hidden):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, trunk_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(trunk_channels),
                nn.ReLU(),
            )
            self.res_tower  = nn.Sequential(
                *[ResidualBlock(trunk_channels) for _ in range(num_res_blocks)]
            )
            self.policy_conv = nn.Sequential(
                nn.Conv2d(trunk_channels, policy_channels, 1, bias=False),
                nn.BatchNorm2d(policy_channels),
                nn.ReLU(),
            )
            self.policy_fc = nn.Conv1d(policy_channels, 1, 1)
            self.value_conv = nn.Sequential(
                nn.Conv2d(trunk_channels, value_channels, 1, bias=False),
                nn.BatchNorm2d(value_channels),
                nn.ReLU(),
            )
            self.value_fc = nn.Sequential(
                nn.Linear(value_channels * rows * cols, value_hidden),
                nn.ReLU(),
                nn.Linear(value_hidden, 1),
                nn.Tanh(),
            )

        def forward(self, x):
            h = self.stem(x)
            h = self.res_tower(h)
            p = self.policy_conv(h).mean(dim=2)
            p = self.policy_fc(p).squeeze(1)
            v = self.value_conv(h).flatten(1)
            v = self.value_fc(v)
            return p, v

        def predict(self, x, legal_mask=None):
            p, v = self.forward(x)
            if legal_mask is not None:
                p = p.masked_fill(~legal_mask, float("-inf"))
            return torch.softmax(p, dim=-1), v

    # ── Game logic ───────────────────────────────────────────────────────────

    def legal_actions(board):
        return [c for c in range(COLS) if board[c] == 0]

    def apply_action(board, col, mark):
        board = list(board)
        for row in range(ROWS - 1, -1, -1):
            if board[row * COLS + col] == 0:
                board[row * COLS + col] = mark
                return tuple(board)
        return None  # column full (shouldn't happen)

    def check_win(board, mark):
        b = np.array(board, dtype=np.int8).reshape(ROWS, COLS)
        # horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                if all(b[r, c+i] == mark for i in range(4)):
                    return True
        # vertical
        for r in range(ROWS - 3):
            for c in range(COLS):
                if all(b[r+i, c] == mark for i in range(4)):
                    return True
        # diagonals
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if all(b[r+i,   c+i] == mark for i in range(4)): return True
                if all(b[r+3-i, c+i] == mark for i in range(4)): return True
        return False

    def is_terminal(board, last_mark):
        if check_win(board, last_mark):
            return True, last_mark
        if all(board[c] != 0 for c in range(COLS)):
            return True, 0   # draw
        return False, None

    def next_mark(mark):
        return 3 - mark

    def board_to_obs(board, mark):
        b = np.array(board, dtype=np.float32).reshape(ROWS, COLS)
        obs = np.zeros((1, 2, ROWS, COLS), dtype=np.float32)
        obs[0, 0] = (b == mark).astype(np.float32)
        obs[0, 1] = (b == next_mark(mark)).astype(np.float32)
        return obs

    # ── Opening book ─────────────────────────────────────────────────────────

    _BOOK_HEIGHT  = 6
    _BOOK_WIDTH   = 7
    _BOOK_MIN_SCORE = -18

    def _book_next_prime(n):
        def _is_prime(x):
            if x < 2: return False
            i = 2
            while i * i <= x:
                if x % i == 0: return False
                i += 1
            return True
        while not _is_prime(n):
            n += 1
        return n

    def _book_partial_key3(key, col, cp, mask):
        pos = 1 << (col * (_BOOK_HEIGHT + 1))
        while pos & mask:
            key *= 3
            key += 1 if (pos & cp) else 2
            pos <<= 1
        return key * 3

    def _book_key3(cp, mask):
        kf = 0
        for c in range(_BOOK_WIDTH):
            kf = _book_partial_key3(kf, c, cp, mask)
        kr = 0
        for c in range(_BOOK_WIDTH - 1, -1, -1):
            kr = _book_partial_key3(kr, c, cp, mask)
        return min(kf, kr) // 3

    def _book_get(key3_val, pk_bytes, size, keys_raw, vals_raw):
        pos = key3_val % size
        stored = int.from_bytes(keys_raw[pos * pk_bytes:(pos + 1) * pk_bytes], 'little')
        if stored == (key3_val & ((1 << (pk_bytes * 8)) - 1)):
            return vals_raw[pos]
        return 0

    def _load_book(data):
        w, h, depth, pk_bytes, v_bytes, log_size = data[:6]
        size = _book_next_prime(1 << log_size)
        off  = 6
        keys = data[off: off + size * pk_bytes]
        vals = data[off + size * pk_bytes: off + size * pk_bytes + size]
        return depth, pk_bytes, size, keys, vals

    def _board_to_bitmask(board, mark):
        opp = 3 - mark
        cp = 0; mask = 0
        for col in range(COLS):
            for rbottom in range(ROWS):
                row_kg = ROWS - 1 - rbottom
                cell = board[row_kg * COLS + col]
                if cell:
                    bit = 1 << (col * (_BOOK_HEIGHT + 1) + rbottom)
                    mask |= bit
                    if cell == mark:
                        cp |= bit
        return cp, mask

    def _best_book_move(depth, pk_bytes, size, keys_raw, vals_raw, board, mark):
        cp, mask = _board_to_bitmask(board, mark)
        nb = bin(mask).count('1')
        if nb >= depth:
            return None
        best_col = None; best_score = None
        for col in range(COLS):
            top_bit = 1 << (col * (_BOOK_HEIGHT + 1) + _BOOK_HEIGHT - 1)
            if mask & top_bit:
                continue
            bottom  = 1 << (col * (_BOOK_HEIGHT + 1))
            col_mask = ((1 << _BOOK_HEIGHT) - 1) << (col * (_BOOK_HEIGHT + 1))
            move_bit = (mask + bottom) & col_mask
            new_cp   = cp ^ mask
            new_mask = mask | move_bit
            raw   = _book_get(_book_key3(new_cp, new_mask), pk_bytes, size, keys_raw, vals_raw)
            if raw == 0:
                continue
            our_score = -( raw + _BOOK_MIN_SCORE - 1 )
            if best_score is None or our_score > best_score:
                best_score = our_score; best_col = col
        return best_col

    # ── MCTS ─────────────────────────────────────────────────────────────────

    class Node:
        __slots__ = ("board", "mark", "prior", "visit_count",
                     "total_value", "children", "is_expanded",
                     "is_terminal", "terminal_value", "parent")

        def __init__(self, board, mark, prior=0.0, parent=None):
            self.board          = board
            self.mark           = mark
            self.prior          = prior
            self.visit_count    = 0
            self.total_value    = 0.0
            self.children       = {{}}   # action -> Node
            self.is_expanded    = False
            self.is_terminal    = False
            self.terminal_value = None
            self.parent         = parent

        @property
        def q_value(self):
            return self.total_value / self.visit_count if self.visit_count else 0.0

        def ucb(self, c_puct, sqrt_parent):
            return -self.q_value + c_puct * self.prior * sqrt_parent / (1 + self.visit_count)

    def expand(node, model, device):
        """Evaluate node with network and expand children."""
        # check terminal
        legal = legal_actions(node.board)
        if not legal:
            node.is_terminal    = True
            node.terminal_value = 0.0
            return 0.0

        # check if previous move caused a win (current player to move means previous player might have won)
        opp = next_mark(node.mark)
        won, winner = is_terminal(node.board, opp)
        if won:
            node.is_terminal    = True
            node.terminal_value = -1.0  # opponent won → bad for current player
            return node.terminal_value

        obs  = torch.from_numpy(board_to_obs(node.board, node.mark)).to(device)
        mask = torch.zeros(1, COLS, dtype=torch.bool, device=device)
        for a in legal:
            mask[0, a] = True

        with torch.no_grad():
            probs, value = model.predict(obs, mask)
        probs  = probs[0].cpu().numpy()
        value  = value[0, 0].item()

        for a in legal:
            child_board = apply_action(node.board, a, node.mark)
            node.children[a] = Node(child_board, next_mark(node.mark),
                                    prior=float(probs[a]), parent=node)
        node.is_expanded = True
        return value

    def select_leaf(root, c_puct):
        node = root
        while node.is_expanded and not node.is_terminal:
            sqrt_v  = math.sqrt(node.visit_count)
            best_a  = max(node.children,
                          key=lambda a: node.children[a].ucb(c_puct, sqrt_v))
            node = node.children[best_a]
        return node

    def backup(node, value):
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value
            node = node.parent

    def mcts_search(board, mark, model, device, num_sims, c_puct, book_priors=None):
        root = Node(board, mark)
        root_value = expand(root, model, device)
        backup(root, root_value)

        if book_priors and root.children:
            for col, prior in book_priors.items():
                if col in root.children:
                    root.children[col].prior = prior

        for _ in range(num_sims - 1):
            leaf  = select_leaf(root, c_puct)
            if leaf.is_terminal:
                backup(leaf, leaf.terminal_value)
            else:
                value = expand(leaf, model, device)
                backup(leaf, value)

        return root

    def pick_action(root, temperature):
        if not root.children:
            return legal_actions(root.board)[0]
        if temperature == 0.0:
            return max(root.children, key=lambda a: root.children[a].visit_count)
        visits = {{a: c.visit_count for a, c in root.children.items()}}
        powered = {{a: v ** (1.0 / temperature) for a, v in visits.items()}}
        total   = sum(powered.values())
        if total == 0:
            return list(visits.keys())[0]
        probs   = [powered[a] / total for a in visits]
        return np.random.choice(list(visits.keys()), p=probs)

    # ── Model + book loading (cached on first call) ──────────────────────────

    if not hasattr(my_agent, "_model"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kw     = NETWORK_KWARGS
        net    = AZNetwork(
            rows=kw["rows"], cols=kw["cols"],
            in_channels=kw["in_channels"], num_actions=kw["num_actions"],
            trunk_channels=kw["trunk_channels"], num_res_blocks=kw["num_res_blocks"],
            policy_channels=kw.get("policy_channels", 16),
            value_channels=kw.get("value_channels", 16),
            value_hidden=kw.get("value_hidden", 64),
        )
        sd = torch.load(io.BytesIO(base64.b64decode(WEIGHTS_B64)),
                        map_location=device, weights_only=True)
        net.load_state_dict(sd)
        net.to(device)
        net.eval()
        my_agent._model  = net
        my_agent._device = device
        if BOOK_B64:
            import base64 as _b64
            my_agent._book = _load_book(_b64.b64decode(BOOK_B64))
        else:
            my_agent._book = None

    model  = my_agent._model
    device = my_agent._device

    board  = tuple(observation.board)
    mark   = observation.mark
    legal  = legal_actions(board)

    if not legal:
        return 0

    # Immediate win / block
    for col in legal:
        b2 = apply_action(board, col, mark)
        if check_win(b2, mark):
            return col
    opp = next_mark(mark)
    for col in legal:
        b2 = apply_action(board, col, opp)
        if check_win(b2, opp):
            return col

    book_priors = None
    if my_agent._book is not None:
        import math as _math
        depth, pk_bytes, size, keys_raw, vals_raw = my_agent._book
        cp, mask = _board_to_bitmask(board, mark)
        nb = bin(mask).count('1')
        if nb < depth:
            book_scores = {{}}
            for col in legal:
                top_bit = 1 << (col * (_BOOK_HEIGHT + 1) + _BOOK_HEIGHT - 1)
                if mask & top_bit:
                    continue
                bottom   = 1 << (col * (_BOOK_HEIGHT + 1))
                col_mask = ((1 << _BOOK_HEIGHT) - 1) << (col * (_BOOK_HEIGHT + 1))
                move_bit = (mask + bottom) & col_mask
                raw = _book_get(_book_key3(cp ^ mask, mask | move_bit),
                                pk_bytes, size, keys_raw, vals_raw)
                if raw != 0:
                    book_scores[col] = -( raw + _BOOK_MIN_SCORE - 1 )
            if book_scores:
                max_s = max(book_scores.values())
                span  = max(max_s - min(book_scores.values()), 1.0)
                exps  = {{col: _math.exp((s - max_s) / span)
                          for col, s in book_scores.items()}}
                total = sum(exps.values())
                book_priors = {{col: v / total for col, v in exps.items()}}

    root = mcts_search(board, mark, model, device, MCTS_SIMS, C_PUCT,
                       book_priors=book_priors)
    return pick_action(root, TEMPERATURE)
'''

with open(OUTPUT, 'w') as f:
    f.write(SUBMISSION)

print(f"\nCreated {OUTPUT}  ({len(SUBMISSION) // 1024} KB)")
print(f"MCTS simulations: {MCTS_SIMS}  |  temperature: {TEMPERATURE}  |  c_puct: {C_PUCT}")

# ── local smoke test ──────────────────────────────────────────────────────────
print("\nRunning smoke test...", flush=True)
ns = {}
exec(compile(SUBMISSION, OUTPUT, 'exec'), ns)
_agent = ns['my_agent']

class _Obs:
    board = [0] * 42
    mark  = 1

class _Cfg:
    rows    = 6
    columns = 7

action = _agent(_Obs(), _Cfg())
print(f"Empty board action: {action}  (expected: 3)  {'✓' if action == 3 else '✗'}")

action2 = _agent(_Obs(), _Cfg())
print(f"Second call (cached): {action2}  ✓")
