import os

import numpy as np
import pytest

import omok
from omok import transforms
from omok.alphazero import AlphaZeroAgent, env_from_state

MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')


def agent(rule='renju', **kwargs):
    return AlphaZeroAgent(rule=rule, model_path=os.path.join(MODELS, 'alphazero-%s.onnx' % rule), seed=0, **kwargs)


def position(moves, rule='renju'):
    env = omok.Omok(rule=rule)
    for m in moves:
        assert env(m) == 0
    return env


# black (1) to move; 113 is forbidden (a double-three: 112-113-114 and 98-113-128)
FORBIDDEN = [112, 0, 114, 14, 98, 210, 128, 224]


def test_env_from_state_matches_the_real_env():
    for moves in ([], [112], FORBIDDEN, FORBIDDEN + [30]):
        env = position(moves)
        rebuilt = env_from_state(env.get_state(), env.get_player())
        assert rebuilt.get_player() == env.get_player()
        assert np.array_equal(rebuilt.get_legal_mask(), env.get_legal_mask())
        obs, real = rebuilt.get_observation(), env.get_observation()
        assert np.array_equal(obs[[0, 1, 2, 4]], real[[0, 1, 2, 4]])  # all planes but the last move
        for pos in np.flatnonzero(env.get_legal_mask())[:5]:  # whose turn it is after a move
            child, real_child = rebuilt.clone(), env.clone()
            child.move(int(pos))
            real_child.move(int(pos))
            assert child.get_player() == real_child.get_player()
    assert position(FORBIDDEN).get_forbidden() == [113]
    with pytest.raises(ValueError):
        env_from_state(position([112]).get_state(), 1)  # white is to move


def test_network_ignores_the_last_move_plane():
    a = agent()
    env = position(FORBIDDEN)
    obs, mask = env.get_observation()[None], env.get_legal_mask()[None]
    moved = obs.copy()
    moved[0, 3] = 0
    moved[0, 3].flat[5] = 1
    p1, v1 = a.evaluate(obs, mask, np.array([0]))
    p2, v2 = a.evaluate(moved, mask, np.array([0]))
    assert np.allclose(p1, p2, atol=1e-6) and np.allclose(v1, v2, atol=1e-6)


@pytest.mark.parametrize('simulations', [0, 32])
def test_probs(simulations):
    a = agent(simulations=simulations)
    env = position(FORBIDDEN)
    state, player = env.get_state(), env.get_player()
    probs = a.get_probs(state, player)
    assert probs.shape == (225,) and abs(probs.sum() - 1) < 1e-5
    assert (probs[state.reshape(-1) != 0] == 0).all()
    assert (probs[env.get_forbidden()] == 0).all()
    move = a(state, player)
    assert env.get_legal_mask()[move]
    if simulations == 0:  # averaged over the symmetries: transforming the board transforms the probs
        for code in range(8):
            probs_t = a.get_probs(transforms.transform_board(state, code), player)
            assert np.abs(probs_t - transforms.transform_board(probs.reshape(15, 15), code).ravel()).max() < 1e-5


def test_search_wins_and_blocks():
    a = agent(simulations=64)
    win = position([112, 0, 113, 1, 114, 2, 115, 30])  # black to move: 111 or 116 wins
    assert a(win.get_state(), win.get_player()) in (111, 116)
    block = position([112, 111, 113, 0, 114, 1, 115])  # white to move: black's four is closed at 111, block at 116
    assert a(block.get_state(), block.get_player()) == 116


@pytest.mark.parametrize('rule,simulations', [('renju', 0), ('freestyle', 0), ('renju', 16)])
def test_plays_legal_games(rule, simulations):
    a = agent(rule, simulations=simulations)
    env = omok.Omok(rule=rule)
    while not env.is_done():
        move = a(env.get_state(), env.get_player())
        assert env.get_legal_mask()[move]
        env(move)


def test_web_ui():
    a = agent(simulations=16)
    game = omok.OmokGame(agent=a, rule='renju')
    status = game.handle('move', {'pos': 112, 'reply': True, 'probs': True})
    assert len(status['moves']) == 2 and abs(sum(status['probs']) - 1) < 1e-3
