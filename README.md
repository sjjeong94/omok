# Omok (오목)
Omok is an open-source Python library for developing AI for Omok (Gomoku) and Connect6.

## Install

```bash
$ pip install omok
$ pip install "omok[fast]"   # optional: numba-compiled Renju rule, ~30x faster
```

With the `fast` extra, Renju forbidden move detection is compiled with [numba](https://numba.pydata.org/)
(the first run takes a few seconds to compile, later runs use the on-disk cache).
Without it, a pure Python implementation is used. Set `OMOK_DISABLE_NUMBA=1` to force the pure Python version.

## Development

This project uses [uv](https://docs.astral.sh/uv/).

```bash
$ uv sync            # create a virtual environment and install dependencies
$ uv run pytest      # run tests
$ uv run python -m omok
$ uv build           # build distribution packages
```


## Usage

Play
```bash
$ python -m omok
```

A web server starts at `http://127.0.0.1:8000` and a browser opens automatically.
The AI is `AlphaZeroAgent` with a 200-simulation search, about 1 s per move on a CPU (see "AlphaZero agent" below).
Click the board to place a stone. Keyboard shortcuts: `A` (AI move), `B` (undo), `Space` (reset), `S` (save game record).
Turn on `Show AI probabilities` (shortcut `P`) to show the AI's move probabilities (%) for the current turn as a heatmap on the board.

```bash
$ python -m omok --simulations 0            # the AlphaZero network alone, no search (default: 200 simulations per move)
$ python -m omok --agent policy             # the previous AI, OmokAgent (--model-index 0 or 1)
$ python -m omok --port 8080 --no-browser   # set the port, don't open a browser
$ python -m omok --no-agent                 # run without the AI
$ python -m omok --game connect6            # Connect6 (no AI)
```

### Rule

The default rule is Renju. Black may not play a double-three (3-3), a double-four (4-4), or an overline (six or more in a row), and wins only with exactly five.
If a move makes a five and a forbidden shape at the same time, the five takes priority. White has no restrictions, and an overline also counts as a win for White.
In the web UI, forbidden points are marked with a red × on Black's turn.

```bash
$ python -m omok --rule freestyle           # freestyle, no forbidden moves
```

```python
env = omok.Omok(rule='freestyle')  # rule='renju' by default
env.get_forbidden()   # Black's current forbidden points (Renju only, on Black's turn; otherwise [])
```

Environment
```python
import omok

env = omok.Omok()
for move in [112, 111, 96, 97, 128, 113, 80, 127, 144]:
    env(move)
print(env)

"""
Result
+-------------------------------+
| - - - - - - - - - - - - - - - |
| - - - - - - - - - - - - - - - |
| - - - - - - - - - - - - - - - |
| - - - - - - - - - - - - - - - |
| - - - - - - - - - - - - - - - |
| - - - - - O - - - - - - - - - |
| - - - - - - O X - - - - - - - |
| - - - - - - X O X - - - - - - |
| - - - - - - - X O - - - - - - |
| - - - - - - - - - O - - - - - |
| - - - - - - - - - - - - - - - |
| - - - - - - - - - - - - - - - |
| - - - - - - - - - - - - - - - |
| - - - - - - - - - - - - - - - |
| - - - - - - - - - - - - - - - |
+-------------------------------+
| Player 2  Winner 1  Moves   9 |
+-------------------------------+
"""
```

Agent
``` python
import omok

agent = omok.OmokAgent(model_index=1)
env = omok.Omok()
while True:
    state = env.get_state()
    player = env.get_player()
    action = agent(state, player)
    env(action)
    print(env)
    if env.is_done():
        break
```

AlphaZero agent
```python
import omok

agent = omok.AlphaZeroAgent(rule='renju', simulations=200)   # simulations=0: the network alone
env = omok.Omok(rule='renju')
while not env.is_done():
    env(agent(env.get_state(), env.get_player()))
probs = agent.get_probs(env.get_state(), env.get_player())   # same interface as OmokAgent
```

`AlphaZeroAgent` plays with a policy-value network trained by self-play (Gumbel AlphaZero) in
[omok-rl](https://github.com/sjjeong94/omok-rl), one network per rule (`alphazero-renju.onnx`, `alphazero-freestyle.onnx`),
with an optional PUCT search (numpy only). Over 100 games from random 4-move openings against `OmokAgent(model_index=1)`:

| Rule | Network alone | 200 simulations |
|---|---:|---:|
| Renju | 0.76 | 0.91 |
| freestyle | 0.82 | 0.88 |

(score: win 1, draw 0.5, loss 0; half of the games as each color). A search of 200 simulations takes about 1 s per move on a CPU.
`get_probs` returns the network's probabilities averaged over the 8 symmetries (`simulations=0`) or the search's visit shares.

Reinforcement Learning
```python
import numpy as np
import omok

env = omok.Omok()
while not env.is_done():
    obs = env.get_observation()   # (5, 15, 15) float32, see below
    mask = env.get_legal_mask()   # (225,) bool: legal moves (excludes Renju forbidden points)
    action = np.random.choice(np.flatnonzero(mask))
    env(action)                   # 0: continue, 1: game over, -1: illegal move
print(env.get_winner())           # 0: draw, 1: black wins, 2: white wins
```

Gymnasium-style `step()`
```python
env = omok.Omok()
obs, info = env.reset()
terminated = False
while not terminated:
    action = np.random.choice(np.flatnonzero(info['action_mask']))
    obs, reward, terminated, truncated, info = env.step(action)
print(reward, info['winner'])  # reward is for the player who just moved (1 for a win, else 0)
```

- `obs` and `info['action_mask']` are for the player to move next; `reward` is for the player who just moved.
- `step()` raises `ValueError` on an illegal move or when the game is already over.
- The Renju forbidden points are cached until the board changes, so repeated `get_legal_mask()` calls are cheap.
- Renju env steps are about 2x slower than freestyle with numba, and about 50-70x slower without it.
- `get_state()` and `get_move_history()` return copies, so they are safe to store as-is.
- A position out of range (outside `0 <= pos < 225`) raises `ValueError`.
- Undo moves with `move_back()`.

Observation planes of `get_observation()`, all from the current player's view:

| Plane | `Omok` (5, 15, 15) | `Connect6` (5, 19, 19) |
|---|---|---|
| 0 | own stones | own stones |
| 1 | opponent stones | opponent stones |
| 2 | 1 if black to play | 1 if black to play |
| 3 | last stone placed | last stone placed |
| 4 | forbidden points (Renju, black to play) | 1 if the next stone is the last one of the turn |

Search (e.g. MCTS)
```python
env = omok.Omok.from_moves([112, 111, 96])   # replay moves; kwargs go to the constructor (rule=...)
child = env.clone()                          # independent copy, ~6x faster than copy.deepcopy
child.move(97)
```

Symmetry augmentation
```python
from omok import transforms

obs = env.get_observation()
policy = np.zeros(225, np.float32)           # e.g. MCTS visit distribution
for obs_t, policy_t in transforms.symmetries(obs, policy):   # 8 rotations/reflections
    ...
```

`transforms.transform_board`, `transform_action` and `transform_policy` apply a single symmetry code (0-7)
and its inverse, and work for any board size.

Connect6
```python
import omok

env = omok.Connect6()   # 19x19, six or more in a row wins
for move in [180, 179, 161, 160, 200, 181, 199, 140, 220, 198, 162, 120]:
    env(move)           # black opens with 1 stone, then each side places 2 per turn
print(env.get_winner())  # 1: black completes six with the last move
```

`Connect6` provides the same interface as `Omok` (`step`, `get_legal_mask`, `get_observation`, `from_moves`, `clone`, `move_back`, etc.).

### License

MIT
