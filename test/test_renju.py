import numpy as np
import omok
from omok import renju


def make_state(black=(), white=()):
    state = np.zeros((15, 15), np.uint8)
    for x, y in black:
        state[y, x] = 1
    for x, y in white:
        state[y, x] = 2
    return state


def pos(x, y):
    return y * 15 + x


def test_double_three():
    state = make_state(black=[(5, 7), (6, 7), (7, 5), (7, 6)])
    assert renju.is_forbidden(state, pos(7, 7))


def test_four_three_allowed():
    state = make_state(black=[(4, 7), (5, 7), (6, 7), (7, 5), (7, 6)])
    assert not renju.is_forbidden(state, pos(7, 7))


def test_double_four():
    state = make_state(black=[(4, 7), (5, 7), (6, 7), (7, 4), (7, 5), (7, 6)],
                       white=[(3, 7), (7, 3)])
    assert renju.is_forbidden(state, pos(7, 7))


def test_double_four_in_line():
    # B . B B [B] . B
    state = make_state(black=[(3, 7), (5, 7), (6, 7), (9, 7)])
    assert renju.is_forbidden(state, pos(7, 7))


def test_overline():
    state = make_state(black=[(2, 7), (3, 7), (4, 7), (6, 7), (7, 7)])
    assert renju.is_forbidden(state, pos(5, 7))


def test_five_has_priority():
    state = make_state(black=[(3, 7), (4, 7), (5, 7), (6, 7),
                              (7, 5), (7, 6), (5, 5), (6, 6)])
    assert not renju.is_forbidden(state, pos(7, 7))


def test_blocked_three_is_not_three():
    # W B B [B] . W  cannot become a straight four
    state = make_state(black=[(5, 7), (6, 7), (7, 5), (7, 6)],
                       white=[(4, 7), (9, 7)])
    assert not renju.is_forbidden(state, pos(7, 7))


def test_three_needs_legal_completion():
    # horizontal . B B [B] . is a three only if its completion is legal.
    # Both completions (4, 7) and (8, 7) are made double-four points.
    black = [(5, 7), (6, 7), (7, 5), (7, 6),
             (4, 9), (4, 10), (4, 11), (1, 7), (8, 9), (8, 10), (8, 11), (11, 7)]
    white = [(4, 12), (8, 12), (0, 7), (12, 7)]
    state = make_state(black=black, white=white)
    state[7, 7] = 1
    assert renju.is_forbidden(state, pos(4, 7))
    assert renju.is_forbidden(state, pos(8, 7))
    state[7, 7] = 0
    assert not renju.is_forbidden(state, pos(7, 7))


def test_env_rejects_forbidden_move():
    env = omok.Omok()
    # black: (5,7) (6,7) (7,5) (7,6), white far away
    for move in [pos(5, 7), pos(0, 0), pos(6, 7), pos(0, 2),
                 pos(7, 5), pos(0, 4), pos(7, 6), pos(0, 6)]:
        assert env(move) == 0
    assert pos(7, 7) in env.get_forbidden()
    assert env(pos(7, 7)) == -1
    assert env.get_player() == 1

    env = omok.Omok(rule='freestyle')
    for move in [pos(5, 7), pos(0, 0), pos(6, 7), pos(0, 2),
                 pos(7, 5), pos(0, 4), pos(7, 6), pos(0, 6)]:
        env(move)
    assert env.get_forbidden() == []
    assert env(pos(7, 7)) == 0


def test_white_overline_wins():
    env = omok.Omok()
    white = [pos(x, 0) for x in (0, 1, 2, 4, 5, 3)]
    black = [pos(x, 14) for x in (0, 2, 4, 6, 8, 10)]
    for b, w in zip(black, white):
        env(b)
        result = env(w)
    assert result == 1
    assert env.get_winner() == 2
