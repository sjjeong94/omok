import numpy as np
import pytest
from omok import renju

numba_impl = pytest.importorskip('omok._renju_numba')


def random_boards(n, seed=0):
    rng = np.random.default_rng(seed)
    for t in range(n):
        k = rng.integers(5, 120)
        if t % 2:  # clustered, like real games
            idx = np.clip(rng.integers(40, 185) + rng.integers(-40, 40, k), 0, 224)
        else:
            idx = rng.choice(225, k, replace=False)
        state = np.zeros(225, np.uint8)
        state[idx] = np.where(rng.random(len(idx)) < rng.uniform(0.4, 0.8), 1, 2)
        yield state.reshape(15, 15)


def test_numba_matches_python():
    total = 0
    for state in random_boards(300):
        expected = renju._py_forbidden_moves(state)
        board = state.reshape(-1).copy()
        assert numba_impl.forbidden_moves(board).tolist() == expected
        assert (board == state.reshape(-1)).all()  # board restored after search
        assert numba_impl.has_legal_move(board) == renju._py_has_legal_move(state)
        for pos in expected[:3]:
            assert numba_impl.is_forbidden(board, pos)
        total += len(expected)
    assert total > 500  # boards actually exercise forbidden shapes


@pytest.mark.skipif(renju.BACKEND != 'numba', reason='OMOK_DISABLE_NUMBA is set')
def test_public_api_uses_numba():
    state = np.zeros((15, 15), np.uint8)
    state.flat[[110, 111, 82, 97]] = 1
    assert renju.forbidden_moves(state) == [112]
    assert all(type(p) is int for p in renju.forbidden_moves(state))
    assert renju.is_forbidden(state, 112) is True
    assert state.flat[112] == 0  # caller's array untouched
