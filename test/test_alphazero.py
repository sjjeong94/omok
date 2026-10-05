import json
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


@pytest.mark.parametrize('simulations', [0, 32])
def test_get_analysis(simulations):
    a = agent(simulations=simulations)
    env = omok.Omok(rule='renju')
    for pos in (112, 113, 127):
        env(pos)
    analysis = a.get_analysis(env.get_state(), env.get_player())
    legal = env.get_legal_mask()
    prior = np.array(analysis['prior'])
    assert abs(prior.sum() - 1) < 1e-3 and not prior[~legal].any()
    assert -1 <= analysis['value'] <= 1
    lines = analysis['lines']
    assert lines and all(legal[line['pos']] and line['pv'][0] == line['pos'] for line in lines)
    if simulations == 0:
        assert 'visits' not in analysis
        return
    visits = np.array(analysis['visits'])
    assert abs(visits.sum() - 1) < 1e-3 and analysis['simulations'] == simulations
    assert [line['visits'] for line in lines] == sorted((line['visits'] for line in lines), reverse=True)
    for line in lines:
        assert line['q'] == analysis['q'][line['pos']] and -1 <= line['q'] <= 1
        sim = env.clone()  # a principal variation is a sequence of legal moves
        for pos in line['pv']:
            assert sim.get_legal_mask()[pos] and not sim.is_done()
            sim(pos)


def test_web_ui_analysis():
    a = agent(simulations=16)
    game = omok.OmokGame(agent=a, rule='renju')
    status = game.handle('move', {'pos': 112, 'analysis': True})
    assert 'analysis' in status and status['analysis']['simulations'] == 16
    json.dumps(status)  # served as JSON


def test_watch():
    a = agent(simulations=48, leaves=8)
    env = position([112, 113])
    frames = []
    assert a.watch(env.get_state(), env.get_player(), lambda f: frames.append(f) or True)
    sims = [f['simulations'] for f in frames]
    assert sims == sorted(sims) and sims[-1] == 48 and len(frames) >= 48 // 8
    legal = env.get_legal_mask()
    for f in frames:
        assert f['paths'] and all(legal[p[0]] for p in f['paths'])
    # the agent keeps the watched search: it plays its most visited move and analyses it as watched
    assert a.get_analysis(env.get_state(), env.get_player())['visits'] == frames[-1]['visits']
    best = max(range(225), key=lambda m: frames[-1]['visits'][m])
    assert frames[-1]['visits'][a(env.get_state(), env.get_player())] == frames[-1]['visits'][best]


def test_watch_stops():
    a = agent(simulations=64, leaves=8)
    env = position([112])
    frames = []
    assert not a.watch(env.get_state(), env.get_player(), lambda f: frames.append(f) or len(frames) < 2)
    assert len(frames) == 2 and frames[-1]['simulations'] < 64
    assert a._cache[0] is None  # an unfinished search is not kept


def test_web_ui_live_search():
    game = omok.OmokGame(agent=agent(simulations=32), rule='renju')
    game.handle('move', {'pos': 112})
    frames, since = [], 0
    while True:
        r = game.handle('search', {'since': since})
        assert r['moves'] == 1 and r['total'] == 32
        frames += r['frames']
        since = len(frames)
        if r['done']:
            break
    assert frames and frames[-1]['simulations'] == 32
    json.dumps(frames)
    status = game.handle('agent', {})  # plays the watched search
    assert status['moves'][-1] in [line['pos'] for line in frames[-1]['lines']]
    game.handle('search', {})  # a new search for the new position...
    game.handle('back', {})  # ...is stopped by a change of position
    assert game.job is None
