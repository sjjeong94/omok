import numpy as np
import pytest
import omok


def pos(x, y):
    return y * 19 + x


def test_turn_order():
    env = omok.Connect6()
    players, stones_left = [], []
    for i in range(7):
        players.append(env.get_player())
        stones_left.append(env.get_stones_left())
        env(i)
    # black 1 stone, then white 2, black 2, white 2 ...
    assert players == [1, 2, 2, 1, 1, 2, 2]
    assert stones_left == [1, 2, 1, 2, 1, 2, 1]


def test_move_back_restores_turn():
    env = omok.Connect6()
    for i in range(4):
        env(i)
    assert env.get_player() == 1 and env.get_stones_left() == 1
    env.move_back()
    assert env.get_player() == 1 and env.get_stones_left() == 2
    env.move_back()
    assert env.get_player() == 2 and env.get_stones_left() == 1
    env.move_back()
    env.move_back()
    assert env.get_player() == 1 and env.get_move_history() == []


def play_row(env, black_x, white_y=18):
    """Black builds a row on y=9 while white plays harmlessly on white_y."""
    white = iter(pos(x, white_y) for x in range(0, 19, 2))
    result = env(pos(black_x[0], 9))
    for i in range(1, len(black_x), 2):
        env(next(white))
        env(next(white))
        result = env(pos(black_x[i], 9))
        if i + 1 < len(black_x):
            result = env(pos(black_x[i + 1], 9))
    return result


def test_six_wins():
    env = omok.Connect6()
    result = play_row(env, [3, 4, 5, 6, 7, 8])
    assert result == 1
    assert env.get_winner() == 1
    assert env.is_done()


def test_five_does_not_win():
    env = omok.Connect6()
    result = play_row(env, [3, 4, 5, 6, 7])
    assert result == 0
    assert env.get_winner() == 0
    assert not env.is_done()


def test_overline_wins():
    env = omok.Connect6()
    result = play_row(env, [3, 4, 5, 7, 8, 9, 6])
    assert result == 1
    assert env.get_winner() == 1


def test_interface():
    env = omok.Connect6()
    assert env.get_state().shape == (19, 19)
    assert env.get_legal_mask().shape == (361,)
    assert env.get_forbidden() == []
    with pytest.raises(ValueError):
        env(361)

    obs = env.get_observation()
    assert obs.shape == (4, 19, 19) and obs.dtype == np.float32
    assert (obs[2] == 1).all() and (obs[3] == 1).all()  # black, single stone
    env(pos(9, 9))
    obs = env.get_observation()
    assert obs[1, 9, 9] == 1 and (obs[2] == 0).all() and (obs[3] == 0).all()
    assert 'Moves   1' in repr(env)


def test_random_games_finish():
    env = omok.Connect6()
    rng = np.random.default_rng(0)
    for _ in range(20):
        env.reset()
        while not env.is_done():
            env(int(rng.choice(np.flatnonzero(env.get_legal_mask()))))
        assert env.get_winner() in (0, 1, 2)
