# Omok (오목)
Omok은 오목 인공지능 개발을 위한 오픈소스 파이썬 라이브러리입니다.

## Install

```bash
$ pip install omok
```


## Usage

Play
```bash
$ python -m omok
```

Environment
```python
import omok

env = omok.Omok()
for move in [112, 111, 96, 97, 128, 113, 80, 127, 144]:
    env(move)
env.show_state()

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
    result = env(action)
    env.show_state()
    if result:
        break
```

### License

MIT
