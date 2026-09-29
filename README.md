# Omok (오목)
Omok은 오목 인공지능 개발을 위한 오픈소스 파이썬 라이브러리입니다.

## Install

```bash
$ pip install omok
```

## Development

[uv](https://docs.astral.sh/uv/)를 사용합니다.

```bash
$ uv sync            # 가상환경 생성 및 의존성 설치
$ uv run pytest      # 테스트
$ uv run python -m omok
$ uv build           # 배포 패키지 빌드
```


## Usage

Play
```bash
$ python -m omok
```

웹 서버가 `http://127.0.0.1:8000` 에서 실행되고 브라우저가 자동으로 열립니다.
보드를 클릭해 착수하고, 단축키 `A`(AI 착수), `B`(무르기), `Space`(리셋), `S`(기보 저장)를 사용할 수 있습니다.
`Show AI probabilities`(단축키 `P`)를 켜면 현재 차례에 대한 AI의 착수 확률(%)이 보드 위에 히트맵으로 표시됩니다.

```bash
$ python -m omok --port 8080 --no-browser   # 포트 지정, 브라우저 자동 실행 끄기
$ python -m omok --no-agent                 # AI 없이 실행
```

### Rule

기본 규칙은 렌주룰(Renju)입니다. 흑은 3-3, 4-4, 장목(6목 이상)에 둘 수 없으며 정확히 5목을 만들어야 승리합니다.
금수와 5목이 동시에 만들어지면 5목이 우선합니다. 백은 제약이 없으며 장목도 승리로 인정됩니다.
웹 UI에서는 흑 차례에 금수 자리가 빨간 × 로 표시됩니다.

```bash
$ python -m omok --rule freestyle           # 금수 없는 자유룰
```

```python
env = omok.Omok(rule='freestyle')
env.get_forbidden()   # 현재 흑의 금수 위치 목록 (렌주룰, 흑 차례에만)
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
    result = env(action)
    print(env)
    if result:
        break
```

### License

MIT
