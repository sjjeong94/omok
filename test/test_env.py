import omok


def test_init():
    env = omok.Omok()
    assert env.state.sum() == 0
    assert env.player == 1
    assert env.winner == 0


def test_reset():
    env = omok.Omok()
    env(1)
    assert env.state.sum() == 1
    assert env.player == 2
    assert env.winner == 0
    env.reset()
    assert env.state.sum() == 0
    assert env.player == 1
    assert env.winner == 0


def test_move():
    env = omok.Omok()
    assert env(1) == 0
    assert env(1) == -1
    assert env(2) == 0


def test_win():
    env = omok.Omok()
    moves = [0, 100, 1, 90, 2, 80, 3, 70, 4]
    for move in moves:
        result = env(move)
    assert result == 1
    assert env.winner == 1
    env.reset()
    moves = [110, 0, 100, 1, 90, 2, 80, 3, 70, 4]
    for move in moves:
        result = env(move)
    assert result == 1
    assert env.winner == 2
