"""Renju forbidden move (3-3, 4-4, overline) detection for black."""

SIZE = 15
EMPTY = 0
BLACK = 1
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


def _inside(x, y):
    return 0 <= x < SIZE and 0 <= y < SIZE


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


def _five_points(board, x, y, dx, dy):
    """Offsets along the line where black completes exactly five through (x, y)."""
    points = []
    for i in range(-4, 5):
        qx, qy = x + dx * i, y + dy * i
        if i == 0 or not _inside(qx, qy) or board[qy * SIZE + qx] != EMPTY:
            continue
        step = 1 if i > 0 else -1
        # stones between (x, y) and q must all be black to share one line
        if any(board[(y + dy * j) * SIZE + x + dx * j] != BLACK
               for j in range(step, i, step)):
            continue
        board[qy * SIZE + qx] = BLACK
        if _line_length(board, qx, qy, dx, dy) == 5:
            points.append(i)
        board[qy * SIZE + qx] = EMPTY
    return points


def _count_fours(board, x, y, dx, dy):
    points = _five_points(board, x, y, dx, dy)
    if len(points) == 2 and points[1] - points[0] == 5:
        return 1  # a straight four is a single four
    return len(points)


def _is_open_four(board, x, y, dx, dy):
    points = _five_points(board, x, y, dx, dy)
    return len(points) == 2 and points[1] - points[0] == 5


def _is_three(board, x, y, dx, dy):
    """A three can become a straight four by a move that is not forbidden."""
    for i in range(-4, 5):
        qx, qy = x + dx * i, y + dy * i
        if i == 0 or not _inside(qx, qy) or board[qy * SIZE + qx] != EMPTY:
            continue
        q = qy * SIZE + qx
        board[q] = BLACK
        open_four = _is_open_four(board, x, y, dx, dy)
        board[q] = EMPTY
        if open_four and not _is_forbidden(board, q):
            return True
    return False


def _has_potential(board, x, y):
    """Cheap filter: any forbidden shape needs at least 4 black stones nearby."""
    n = 0
    for dx, dy in DIRECTIONS:
        for i in (-4, -3, -2, -1, 1, 2, 3, 4):
            cx, cy = x + dx * i, y + dy * i
            if _inside(cx, cy) and board[cy * SIZE + cx] == BLACK:
                n += 1
    return n >= 4


def _is_forbidden(board, pos):
    y, x = divmod(pos, SIZE)
    if board[pos] != EMPTY or not _has_potential(board, x, y):
        return False

    board[pos] = BLACK
    try:
        lengths = [_line_length(board, x, y, dx, dy) for dx, dy in DIRECTIONS]
        if 5 in lengths:
            return False  # five takes priority over any forbidden shape
        if max(lengths) > 5:
            return True

        fours = [_count_fours(board, x, y, dx, dy) for dx, dy in DIRECTIONS]
        if sum(fours) >= 2:
            return True

        threes = 0
        for (dx, dy), four in zip(DIRECTIONS, fours):
            if not four and _is_three(board, x, y, dx, dy):
                threes += 1
                if threes >= 2:
                    return True
        return False
    finally:
        board[pos] = EMPTY


def is_forbidden(state, pos):
    """Whether black playing at pos is a forbidden move. state: (15, 15) array."""
    return _is_forbidden(state.reshape(-1).tolist(), int(pos))


def forbidden_moves(state):
    """All empty positions where black is not allowed to play."""
    board = state.reshape(-1).tolist()
    return [pos for pos in range(SIZE * SIZE) if _is_forbidden(board, pos)]


def has_legal_move(state):
    """Whether black has at least one empty position that is not forbidden."""
    board = state.reshape(-1).tolist()
    return any(v == EMPTY and not _is_forbidden(board, pos)
               for pos, v in enumerate(board))
