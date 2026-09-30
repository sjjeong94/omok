"""Board symmetries (the 8 elements of the dihedral group) for data augmentation.

Boards can be any array whose last two axes are a square (size, size) board,
e.g. a (size, size) state or a (C, size, size) observation. Actions are flat
indices y * size + x, and policies are (..., size * size) vectors over them.

A symmetry is a code in [0, 8): bit 0 flips horizontally, bit 1 flips
vertically and bit 2 transposes, applied in that order.
"""
import numpy as np

NUM_SYMMETRIES = 8


def _hflip(board):
    return board[..., :, ::-1]


def _vflip(board):
    return board[..., ::-1, :]


def _transpose(board):
    return np.swapaxes(board, -1, -2)


def _ops(code, inverse):
    ops = [op for bit, op in enumerate((_hflip, _vflip, _transpose)) if (code >> bit) & 1]
    return ops[::-1] if inverse else ops  # each op is its own inverse


def transform_board(board, code, inverse=False):
    """Apply symmetry code to the last two axes of board. Returns a new array."""
    board = np.asarray(board)
    for op in _ops(code, inverse):
        board = op(board)
    return np.ascontiguousarray(board)


def transform_action(action, size, code, inverse=False):
    """Map a flat action index through symmetry code on a size x size board."""
    y, x = divmod(int(action), size)
    for op in _ops(code, inverse):
        if op is _hflip:
            x = size - 1 - x
        elif op is _vflip:
            y = size - 1 - y
        else:
            x, y = y, x
    return y * size + x


def transform_policy(policy, code, inverse=False):
    """Apply symmetry code to (..., size * size) policy vectors."""
    policy = np.asarray(policy)
    size = int(round(np.sqrt(policy.shape[-1])))
    if size * size != policy.shape[-1]:
        raise ValueError('policy length %d is not a square board' % policy.shape[-1])
    board = policy.reshape(policy.shape[:-1] + (size, size))
    return transform_board(board, code, inverse).reshape(policy.shape)


def symmetries(obs, policy):
    """All 8 symmetric (obs, policy) pairs, starting with the original.

    obs: (..., size, size) observation, policy: (..., size * size) target."""
    return [(transform_board(obs, code), transform_policy(policy, code))
            for code in range(NUM_SYMMETRIES)]


class HorizontalFlip:
    def __call__(self, state, action):
        size = np.shape(state)[-1]
        return transform_board(state, 1), transform_action(action, size, 1)


class VerticalFlip:
    def __call__(self, state, action):
        size = np.shape(state)[-1]
        return transform_board(state, 2), transform_action(action, size, 2)


class Transpose:
    def __call__(self, state, action):
        size = np.shape(state)[-1]
        return transform_board(state, 4), transform_action(action, size, 4)
