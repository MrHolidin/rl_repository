# Запустить в Kaggle notebook (где есть доступ к model_weights.pt)

import torch
import base64
import io

# === НАСТРОЙКИ ===
MINIMAX_DEPTH = 2  # Глубина поиска (0 = только DQN, 1+ = minimax с DQN как heuristic)

# 1. Загрузить веса
weights_path = '/kaggle/input/dqn1/pytorch/v2/1/model_weights.pt'
sd = torch.load(weights_path, map_location='cpu')

# 2. Сериализовать в base64
buffer = io.BytesIO()
torch.save(sd, buffer)
weights_b64 = base64.b64encode(buffer.getvalue()).decode('ascii')

# 3. Сгенерировать submission.py с встроенными весами
submission_code = '''"""Kaggle Connect X submission - DQN + Minimax agent with embedded weights."""

def my_agent(observation, configuration):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import base64
    import io
    
    WEIGHTS_B64 = "''' + weights_b64 + '''"
    DEPTH = ''' + str(MINIMAX_DEPTH) + '''
    
    class ResidBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        def forward(self, x):
            h = F.relu(self.conv1(x))
            h = self.conv2(h)
            return F.relu(x + h)

    class Connect4QRDQN(nn.Module):
        def __init__(self):
            super().__init__()
            trunk_channels, adv_hidden, val_hidden, n_quantiles = 96, 64, 128, 32
            self.stem = nn.Conv2d(4, trunk_channels, kernel_size=3, padding=1)
            self.res_blocks = nn.ModuleList([ResidBlock(trunk_channels) for _ in range(2)])
            self.adv1 = nn.Conv1d(trunk_channels, adv_hidden, kernel_size=1)
            self.adv2_quantile = nn.Conv1d(adv_hidden, n_quantiles, kernel_size=1)
            self.val1 = nn.Linear(trunk_channels, val_hidden)
            self.val2_quantile = nn.Linear(val_hidden, n_quantiles)

        def forward(self, x):
            b, _, r, c = x.shape
            rr = torch.linspace(-1, 1, steps=r, device=x.device).view(1, 1, r, 1).expand(b, 1, r, c)
            cc = torch.linspace(-1, 1, steps=c, device=x.device).view(1, 1, 1, c).expand(b, 1, r, c)
            x = torch.cat([x, rr, cc], dim=1)
            h = F.relu(self.stem(x))
            for blk in self.res_blocks:
                h = blk(h)
            h_col = h.mean(dim=2)
            a = F.relu(self.adv1(h_col))
            a = self.adv2_quantile(a).permute(0, 2, 1)
            h_global = h.mean(dim=(2, 3))
            v = F.relu(self.val1(h_global))
            v = self.val2_quantile(v).unsqueeze(1)
            return v + (a - a.mean(dim=1, keepdim=True))

    def get_legal(board, cols):
        return [c for c in range(cols) if board[c] == 0]
    
    def drop_piece(board, col, mark):
        board = list(board)
        for row in range(5, -1, -1):
            if board[row * 7 + col] == 0:
                board[row * 7 + col] = mark
                return board
        return None
    
    def check_win(board, mark):
        b = np.array(board).reshape(6, 7)
        # Horizontal
        for r in range(6):
            for c in range(4):
                if all(b[r, c+i] == mark for i in range(4)):
                    return True
        # Vertical
        for r in range(3):
            for c in range(7):
                if all(b[r+i, c] == mark for i in range(4)):
                    return True
        # Diagonals
        for r in range(3):
            for c in range(4):
                if all(b[r+i, c+i] == mark for i in range(4)):
                    return True
                if all(b[r+3-i, c+i] == mark for i in range(4)):
                    return True
        return False
    
    def evaluate_dqn(board, mark, model, device):
        board_np = np.array(board, dtype=np.float32).reshape(6, 7)
        obs = np.zeros((1, 2, 6, 7), dtype=np.float32)
        obs[0, 0] = (board_np == mark)
        obs[0, 1] = (board_np == (3 - mark))
        x = torch.from_numpy(obs).to(device)
        with torch.no_grad():
            q = model(x).mean(dim=-1).squeeze(0)
        return q.max().item()
    
    def negamax(board, mark, depth, alpha, beta, model, device, cols):
        legal = get_legal(board, cols)
        if not legal:
            return 0.0
        
        opp = 3 - mark
        if check_win(board, opp):
            return -1.0
        
        if depth == 0:
            return evaluate_dqn(board, mark, model, device)
        
        value = -float('inf')
        for col in legal:
            new_board = drop_piece(board, col, mark)
            if new_board is None:
                continue
            if check_win(new_board, mark):
                return 1.0
            child_val = -negamax(new_board, opp, depth - 1, -beta, -alpha, model, device, cols)
            value = max(value, child_val)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    if not hasattr(my_agent, '_model'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Connect4QRDQN()
        weights_bytes = base64.b64decode(WEIGHTS_B64)
        buffer = io.BytesIO(weights_bytes)
        sd = torch.load(buffer, map_location=device, weights_only=True)
        model.load_state_dict(sd)
        model.to(device)
        model.eval()
        my_agent._model = model
        my_agent._device = device

    model = my_agent._model
    device = my_agent._device
    
    board = observation.board
    mark = observation.mark
    cols = configuration.columns
    
    legal_actions = get_legal(board, cols)
    if not legal_actions:
        return 0
    
    if DEPTH == 0:
        board_np = np.array(board, dtype=np.float32).reshape(6, 7)
        obs = np.zeros((1, 2, 6, 7), dtype=np.float32)
        obs[0, 0] = (board_np == mark)
        obs[0, 1] = (board_np == (3 - mark))
        x = torch.from_numpy(obs).to(device)
        with torch.no_grad():
            q_values = model(x).mean(dim=-1).squeeze(0).cpu().numpy()
        return int(max(legal_actions, key=lambda a: q_values[a]))
    
    best_action = legal_actions[0]
    best_value = -float('inf')
    opp = 3 - mark
    
    for col in legal_actions:
        new_board = drop_piece(board, col, mark)
        if new_board is None:
            continue
        if check_win(new_board, mark):
            return col
        value = -negamax(new_board, opp, DEPTH - 1, -float('inf'), float('inf'), model, device, cols)
        if value > best_value:
            best_value = value
            best_action = col
    
    return best_action
'''

# 4. Записать в файл
with open('submission.py', 'w') as f:
    f.write(submission_code)

print(f"Created submission.py ({len(submission_code)} bytes)")
print(f"Minimax depth: {MINIMAX_DEPTH}")

# 5. Тест
exec(submission_code)

class Obs:
    board = [0]*42
    mark = 1
class Cfg:
    columns = 7

action = my_agent(Obs(), Cfg())
print(f"Empty board action: {action} (expected: 3)")

# 6. Тест против negamax (опционально)
# from kaggle_environments import evaluate
# results = evaluate("connectx", [my_agent, "negamax"], num_episodes=10)
# wins = sum(1 for r in results if r[0] > r[1])
# print(f"vs negamax: {wins}/10 wins")