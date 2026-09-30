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


def test_reset_returns_observation():
    env = omok.Omok()
    env(112)
    obs, info = env.reset()
    assert obs.shape == (3, 15, 15) and obs.sum() == 225  # only the black-to-play plane
    assert info['player'] == 1 and info['winner'] == 0
    assert info['action_mask'].all()


def test_step():
    env = omok.Omok()
    obs, reward, terminated, truncated, info = env.step(112)
    assert obs[1, 7, 7] == 1  # white to play, black stone is the opponent's
    assert (reward, terminated, truncated) == (0.0, False, False)
    assert info['player'] == 2 and not info['action_mask'][112]


def test_step_reward_is_for_mover():
    for moves, winner in [([0, 100, 1, 90, 2, 80, 3, 70, 4], 1),
                          ([110, 0, 100, 1, 90, 2, 80, 3, 70, 4], 2)]:
        env = omok.Omok()
        for move in moves[:-1]:
            _, reward, terminated, _, _ = env.step(move)
            assert reward == 0.0 and not terminated
        _, reward, terminated, _, info = env.step(moves[-1])
        assert reward == 1.0 and terminated
        assert info['winner'] == winner and not info['action_mask'].any()


def test_step_errors():
    env = omok.Omok()
    env.step(112)
    with pytest.raises(ValueError):
        env.step(112)  # occupied
    with pytest.raises(ValueError):
        env.step(225)  # out of range
    env.reset()
    for move in [110, 0, 111, 2, 82, 4, 97, 6]:
        env.step(move)
    with pytest.raises(ValueError):
        env.step(112)  # 3-3 forbidden for black
    assert env.get_move_history() == [110, 0, 111, 2, 82, 4, 97, 6]
    env.reset()
    for move in [0, 100, 1, 90, 2, 80, 3, 70, 4]:
        env.step(move)
    with pytest.raises(ValueError):
        env.step(50)  # game over


def test_forbidden_cache_follows_board():
    env = omok.Omok()
    for move in [110, 0, 111, 2, 82, 4, 97, 6]:
        env(move)
    assert 112 in env.get_forbidden() and env.is_forbidden(112)
    env.get_forbidden().append(1)  # returned list is a copy
    assert 1 not in env.get_forbidden()
    env.move_back()  # white's turn: no forbidden moves
    assert env.get_forbidden() == []
    env(6)
    assert 112 in env.get_forbidden()
    env.move_back()
    env.move_back()
    env(113)  # black's (7,6) removed -> (7,7) no longer 3-3
    env(6)
    assert 112 not in env.get_forbidden() and not env.is_forbidden(112)
