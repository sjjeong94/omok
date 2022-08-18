import omok


def test_init():
    env = omok.Omok()
    assert env.get_state().sum() == 0
    assert env.get_player() == 1
    assert env.get_winner() == 0


def test_reset():
    env = omok.Omok()
    env(1)
    assert env.get_state().sum() == 1
    assert env.get_player() == 2
    assert env.get_winner() == 0
    env.reset()
    assert env.get_state().sum() == 0
    assert env.get_player() == 1
    assert env.get_winner() == 0


def test_move():
    env = omok.Omok()
    assert env(1) == 0
    assert env(1) == -1
    assert env(2) == 0


def test_move_back():
    env = omok.Omok()
    for i in range(10):
        env(i)
    for i in range(7):
        env.move_back()
    assert env.get_move_history() == [0, 1, 2]

    env.reset()
    moves = [0, 100, 1, 90, 2, 80, 3, 70, 4]
    for move in moves:
        env(move)
    env.move_back()
    assert env.get_winner() == 0


def test_win():
    env = omok.Omok()
    moves = [0, 100, 1, 90, 2, 80, 3, 70, 4]
    for move in moves:
        result = env(move)
    move_history = env.get_move_history()
    assert result == 1
    assert env.get_winner() == 1
    assert len(moves) == len(move_history)
    for i in range(len(moves)):
        assert moves[i] == move_history[i]

    env.reset()
    moves = [110, 0, 100, 1, 90, 2, 80, 3, 70, 4]
    for move in moves:
        result = env(move)
    move_history = env.get_move_history()
    assert result == 1
    assert env.get_winner() == 2
    assert len(moves) == len(move_history)
    for i in range(len(moves)):
        assert moves[i] == move_history[i]

    env.reset()
    for move in range(15*15):
        result = env(move)
        if result:
            break
    assert result == 1
    assert env.get_winner() == 1
    result = env(224)
    assert result == -1
