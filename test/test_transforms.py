import numpy as np
import pytest
import omok
from omok import transforms


@pytest.mark.parametrize('size', [15, 19])
def test_board_and_action_agree(size):
    rng = np.random.default_rng(0)
    for action in rng.integers(0, size * size, 20):
        board = np.zeros((size, size))
        board.flat[action] = 1
        for code in range(8):
            t = transforms.transform_board(board, code)
            assert t.flat[transforms.transform_action(action, size, code)] == 1
            assert transforms.transform_action(
                transforms.transform_action(action, size, code), size, code, inverse=True) == action


def test_eight_distinct_symmetries_and_inverse():
    board = np.arange(15 * 15).reshape(15, 15)
    outs = [transforms.transform_board(board, code) for code in range(8)]
    assert len({o.tobytes() for o in outs}) == 8
    for code, out in enumerate(outs):
        assert (transforms.transform_board(out, code, inverse=True) == board).all()


@pytest.mark.parametrize('cls', [omok.Omok, omok.Connect6])
def test_symmetries_keep_obs_and_policy_aligned(cls):
    env = cls.from_moves([0, 1, 2, 30, 45])
    obs = env.get_observation()
    policy = env.get_legal_mask().astype(np.float32)
    pairs = transforms.symmetries(obs, policy)
    assert len(pairs) == 8
    assert (pairs[0][0] == obs).all() and (pairs[0][1] == policy).all()
    for o, p in pairs:
        assert o.shape == obs.shape and p.shape == policy.shape
        # legal points are exactly the empty points, in every symmetry
        empty = (o[0] == 0) & (o[1] == 0)
        if cls is omok.Connect6 or env.get_player() == 2:
            assert (empty.reshape(-1) == (p == 1)).all()
        assert o.flags['C_CONTIGUOUS']


def test_policy_batch():
    policy = np.random.default_rng(0).random((4, 225))
    out = transforms.transform_policy(policy, 5)
    for i in range(4):
        assert (out[i] == transforms.transform_policy(policy[i], 5)).all()
    with pytest.raises(ValueError):
        transforms.transform_policy(np.zeros(10), 1)


def test_legacy_classes():
    state = np.zeros((15, 15))
    state[2, 3] = 1  # action 33
    for op in [transforms.HorizontalFlip(), transforms.VerticalFlip(), transforms.Transpose()]:
        s, a = op(state, 33)
        assert s.flat[a] == 1
    s, a = transforms.HorizontalFlip()(state, 33)
    assert a == 2 * 15 + 11 and s[2, 11] == 1
