import numpy as np
import pytest
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

    env = omok.Omok(rule='freestyle')  # black makes 4-4 at (3, 3) in renju
    for move in range(15*15):
        result = env(move)
        if result:
            break
    assert result == 1
    assert env.get_winner() == 1
    result = env(224)
    assert result == -1


def test_state_is_copy():
    env = omok.Omok()
    state = env.get_state()
    history = env.get_move_history()
    env(112)
    assert state.sum() == 0
    assert history == []
    env.get_state()[0, 0] = 2
    env.get_move_history().append(0)
    assert env.get_state()[0, 0] == 0
    assert env.get_move_history() == [112]


def test_invalid_pos():
    env = omok.Omok()
    for pos in [-1, 225, 1000]:
        with pytest.raises(ValueError):
            env(pos)
    assert env.get_move_history() == []


def test_done():
    env = omok.Omok()
    assert not env.is_done()
    for move in [0, 100, 1, 90, 2, 80, 3, 70, 4]:
        env(move)
    assert env.is_done()
    assert env(50) == -1
    assert not env.get_legal_mask().any()
    env.move_back()
    assert not env.is_done()
    env.reset()
    assert not env.is_done()


def test_legal_mask():
    env = omok.Omok()
    mask = env.get_legal_mask()
    assert mask.shape == (225,) and mask.dtype == bool and mask.all()

    # black: (5,7) (6,7) (7,5) (7,6) -> (7,7) is 3-3
    for move in [110, 0, 111, 2, 82, 4, 97, 6]:
        env(move)
    mask = env.get_legal_mask()
    assert not mask[112]
    assert not mask[[110, 0, 111, 2, 82, 4, 97, 6]].any()
    assert mask.sum() == 225 - 8 - len(env.get_forbidden())

    env(113)  # black plays elsewhere, white has no forbidden moves
    assert env.get_legal_mask()[112]


def test_observation():
    env = omok.Omok()
    env(112)
    env(113)
    obs = env.get_observation()
    assert obs.shape == (3, 15, 15) and obs.dtype == np.float32
    assert obs[0, 7, 7] == 1 and obs[1, 7, 8] == 1  # black to play
    assert obs[0].sum() == 1 and obs[1].sum() == 1
    assert (obs[2] == 1).all()
    env(114)
    obs = env.get_observation()
    assert obs[0, 7, 8] == 1 and obs[1].sum() == 2  # white to play
    assert (obs[2] == 0).all()
