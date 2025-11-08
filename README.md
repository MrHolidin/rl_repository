# Connect Four RL Project

Проект для обучения RL-модели игре в крестики-нолики 4 в ряд (Connect Four).

## Описание

Этот проект реализует обучение агента с подкреплением для игры в Connect Four (6×7). Поддерживаются два подхода:
- **Q-learning** (табличный метод)
- **DQN** (Deep Q-Network)

## Структура проекта

```
tic_tac_toe_4inrow_rl/
├── README.md
├── requirements.txt
├── configs/
│   ├── dqn_default.yaml
│   └── qlearning_tabular.yaml
├── data/
│   ├── checkpoints/
│   └── logs/
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   └── connect4_env.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── random_agent.py
│   │   ├── heuristic_agent.py
│   │   ├── qlearning_agent.py
│   │   └── dqn_agent.py
│   ├── models/
│   │   └── dqn_network.py
│   ├── training/
│   │   ├── train_qlearning.py
│   │   ├── train_dqn.py
│   │   └── eval_agent.py
│   ├── utils/
│   │   ├── replay_buffer.py
│   │   ├── metrics.py
│   │   └── serialization.py
│   └── cli/
│       ├── play_human_vs_agent.py
│       └── play_agent_vs_agent.py
└── tests/
    ├── test_env.py
    ├── test_agents.py
    └── test_utils.py
```

## Установка

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd RL
```

2. Создайте виртуальное окружение (рекомендуется):
```bash
# Установите python3-venv если еще не установлен:
# sudo apt install python3-venv

# Создайте виртуальное окружение:
python3 -m venv venv

# Активируйте виртуальное окружение:
source venv/bin/activate
```

3. Установите зависимости:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Примечание:** Если вы не можете использовать виртуальное окружение, можно установить зависимости глобально с флагом `--break-system-packages` (не рекомендуется):
```bash
pip install --break-system-packages -r requirements.txt
```

## Использование

### Обучение Q-learning агента

```bash
python -m src.training.train_qlearning \
    --num-episodes 10000 \
    --learning-rate 0.1 \
    --discount-factor 0.99 \
    --epsilon 0.1 \
    --epsilon-decay 0.995 \
    --epsilon-min 0.01 \
    --eval-freq 100 \
    --eval-episodes 100 \
    --save-freq 1000 \
    --checkpoint-dir data/checkpoints \
    --log-dir data/logs \
    --seed 42
```

### Обучение DQN агента

```bash
python -m src.training.train_dqn \
    --num-episodes 10000 \
    --learning-rate 0.001 \
    --discount-factor 0.99 \
    --epsilon 1.0 \
    --epsilon-decay 0.995 \
    --epsilon-min 0.01 \
    --batch-size 32 \
    --replay-buffer-size 10000 \
    --target-update-freq 100 \
    --eval-freq 100 \
    --eval-episodes 100 \
    --save-freq 1000 \
    --checkpoint-dir data/checkpoints \
    --log-dir data/logs \
    --device cuda \
    --seed 42
```

### Оценка агента

```bash
python -m src.training.eval_agent \
    --agent-path data/checkpoints/dqn_final.pt \
    --agent-type dqn \
    --opponent-type random \
    --num-episodes 1000 \
    --seed 42 \
    --plot \
    --plot-path results.png
```

### Игра против агента

```bash
python -m src.cli.play_human_vs_agent \
    --agent-path data/checkpoints/dqn_final.pt \
    --agent-type dqn \
    --human-first
```

### Игра агент против агента

```bash
python -m src.cli.play_agent_vs_agent \
    --agent1-path data/checkpoints/dqn_final.pt \
    --agent1-type dqn \
    --agent2-type random \
    --num-games 10
```

## Компоненты

### Environment (Connect4Env)

Реализует игру Connect Four:
- Поле 6×7
- Игроки по очереди роняют фишки в столбец
- Победа: 4 в ряд (горизонталь, вертикаль, диагональ)
- Награды: +1 за победу, -1 за проигрыш, 0 за ничью

### Агенты

- **RandomAgent**: Случайный выбор действий
- **HeuristicAgent**: Эвристический агент (пытается выиграть, блокирует противника)
- **QLearningAgent**: Табличный Q-learning
- **DQNAgent**: Deep Q-Network с experience replay

### Training

- Self-play обучение
- Периодическая оценка против фиксированного оппонента
- Логирование метрик в CSV
- Сохранение чекпоинтов

## Метрики

Во время обучения логируются:
- Win rate против оппонента
- Draw rate
- Loss rate
- Episode length
- Epsilon (для epsilon-greedy)
- Replay buffer size (для DQN)

## Тестирование

```bash
pytest tests/
```

## Лицензия

MIT
