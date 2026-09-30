"""Numba-compiled Renju forbidden move detection.

Same algorithm as the pure Python version in omok.renju, which stays the
reference implementation. board is a flat uint8 array of length 225 and is
modified temporarily during the search, so callers must pass a copy.
"""
from numba import njit
import numpy as np

SIZE = 15
EMPTY = 0
BLACK = 1
DX = (1, 0, 1, 1)
DY = (0, 1, 1, -1)


@njit(cache=True)
def _inside(x, y):
    return 0 <= x < SIZE and 0 <= y < SIZE


@njit(cache=True)
def _line_length(board, x, y, dx, dy):
    """Length of the contiguous black line through (x, y)."""
    n = 1
    for s in (1, -1):
        cx, cy = x + dx * s, y + dy * s
        while _inside(cx, cy) and board[cy * SIZE + cx] == BLACK:
            n += 1
            cx += dx * s
            cy += dy * s
    return n


@njit(cache=True)
def _five_points(board, x, y, dx, dy):
    """Number, first and last offsets of points completing exactly five through (x, y)."""
    count, first, last = 0, 0, 0
    for i in range(-4, 5):
        qx, qy = x + dx * i, y + dy * i
        if i == 0 or not _inside(qx, qy) or board[qy * SIZE + qx] != EMPTY:
            continue
        step = 1 if i > 0 else -1
        # stones between (x, y) and q must all be black to share one line
        connected = True
        for j in range(step, i, step):
            if board[(y + dy * j) * SIZE + x + dx * j] != BLACK:
                connected = False
                break
        if not connected:
            continue
        board[qy * SIZE + qx] = BLACK
        if _line_length(board, qx, qy, dx, dy) == 5:
            if count == 0:
                first = i
            last = i
            count += 1
        board[qy * SIZE + qx] = EMPTY
    return count, first, last


@njit(cache=True)
def _count_fours(board, x, y, dx, dy):
    count, first, last = _five_points(board, x, y, dx, dy)
    if count == 2 and last - first == 5:
        return 1  # a straight four is a single four
    return count


@njit(cache=True)
def _is_open_four(board, x, y, dx, dy):
    count, first, last = _five_points(board, x, y, dx, dy)
    return count == 2 and last - first == 5


@njit(cache=True)
def _has_potential(board, x, y):
    """Cheap necessary condition for a forbidden move, see omok.renju."""
    lines = 0
    for d in range(4):
        dx, dy = DX[d], DY[d]
        n = 0
        for s in (1, -1):
            for i in range(1, 5):
                cx, cy = x + dx * i * s, y + dy * i * s
                if not _inside(cx, cy):
                    break
                v = board[cy * SIZE + cx]
                if v == BLACK:
                    n += 1
                elif v != EMPTY:
                    break
        if n >= 4:
            return True
        if n >= 2:
            lines += 1
            if lines >= 2:
                return True
    return False


@njit(cache=True)
def _is_forbidden(board, pos):
    y, x = pos // SIZE, pos % SIZE
    if board[pos] != EMPTY or not _has_potential(board, x, y):
        return False

    board[pos] = BLACK
    result = False
    five = False
    overline = False
    for d in range(4):
        n = _line_length(board, x, y, DX[d], DY[d])
        if n == 5:
            five = True
        elif n > 5:
            overline = True

    if five:
        result = False  # five takes priority over any forbidden shape
    elif overline:
        result = True
    else:
        fours = 0
        has_four = [False, False, False, False]
        for d in range(4):
            c = _count_fours(board, x, y, DX[d], DY[d])
            fours += c
            has_four[d] = c > 0
        if fours >= 2:
            result = True
        else:
            # a three can become a straight four by a move that is not forbidden;
            # inlined here because numba handles self-recursion but not mutual recursion
            threes = 0
            for d in range(4):
                if has_four[d]:
                    continue
                dx, dy = DX[d], DY[d]
                for i in range(-3, 4):
                    qx, qy = x + dx * i, y + dy * i
                    if i == 0 or not _inside(qx, qy) or board[qy * SIZE + qx] != EMPTY:
                        continue
                    q = qy * SIZE + qx
                    board[q] = BLACK
                    open_four = (_line_length(board, x, y, dx, dy) == 4
                                 and _is_open_four(board, x, y, dx, dy))
                    board[q] = EMPTY
                    if open_four and not _is_forbidden(board, q):
                        threes += 1
                        break
                if threes >= 2:
                    result = True
                    break

    board[pos] = EMPTY
    return result


@njit(cache=True)
def is_forbidden(board, pos):
    return _is_forbidden(board, pos)


@njit(cache=True)
def forbidden_moves(board):
    out = np.empty(SIZE * SIZE, np.int64)
    n = 0
    for pos in range(SIZE * SIZE):
        if _is_forbidden(board, pos):
            out[n] = pos
            n += 1
    return out[:n]


@njit(cache=True)
def has_legal_move(board):
    for pos in range(SIZE * SIZE):
        if board[pos] == EMPTY and not _is_forbidden(board, pos):
            return True
    return False
