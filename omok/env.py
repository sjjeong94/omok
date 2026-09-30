import numpy as np
from omok import renju

WIN = 5
SIZE = 15
PLAYER_NONE = 0
PLAYER_BLACK = 1
PLAYER_WHITE = 2
RULES = ('renju', 'freestyle')


def check_match(state, player, action_x, action_y, sx, sy, win=WIN):
    size = state.shape[0]
    match = 1
    for i in range(win):
        y = action_y+(i+1)*sy
        x = action_x+(i+1)*sx
        if (x < 0) or (x >= size) or (y < 0) or (y >= size):
            break
        check = state[y, x]
        if (check == player):
            match += 1
        else:
            break
    for i in range(win):
        y = action_y-(i+1)*sy
        x = action_x-(i+1)*sx
        if (x < 0) or (x >= size) or (y < 0) or (y >= size):
            break
        check = state[y, x]
        if (check == player):
            match += 1
        else:
            break
    return match


class BoardGame:
    """Two-player stone-connecting game: `win` in a row on a `size` x `size` board."""
    size = SIZE
    win = WIN

    def __init__(self):
        self.reset()

    def reset(self):
        self._state = np.zeros(self.size * self.size, np.uint8)
        self._player = PLAYER_BLACK
        self._winner = PLAYER_NONE
        self._done = False
        self._move_history = []
        self._cache = {}
        return self.get_observation(), self._info()

    def _board(self):
        return self._state.reshape(self.size, self.size)

    def _next_player(self, num_moves):
        """Player to move after num_moves stones have been placed."""
        return PLAYER_BLACK if num_moves % 2 == 0 else PLAYER_WHITE

    def get_state(self):
        return self._board().copy()

    def get_player(self):
        return self._player

    def get_winner(self):
        return self._winner

    def is_done(self):
        return self._done

    def get_move_history(self):
        return list(self._move_history)

    def get_stones_left(self):
        """Stones the current player still places in this turn."""
        if self._done:
            return 0
        n = len(self._move_history)
        k = 1
        while self._next_player(n + k) == self._player:
            k += 1
        return k

    def is_forbidden(self, pos):
        return False

    def get_forbidden(self):
        return []

    def get_legal_mask(self):
        """Boolean mask of shape (size*size,) for the moves the current player can make."""
        if self._done:
            return np.zeros(self.size * self.size, bool)
        mask = self._state == PLAYER_NONE
        mask[self.get_forbidden()] = False
        return mask

    def get_observation(self):
        """Float32 planes of shape (3, size, size) from the current player's view:
        own stones, opponent stones, and a constant plane that is 1 if black to play."""
        board = self._board()
        player = self._player
        return np.stack([
            board == player,
            board == player ^ 3,
            np.full(board.shape, player == PLAYER_BLACK),
        ]).astype(np.float32)

    def _info(self):
        return {'player': self._player, 'winner': self._winner,
                'action_mask': self.get_legal_mask()}

    def step(self, action):
        """Gymnasium-style step: returns (obs, reward, terminated, truncated, info).

        obs and info['action_mask'] are for the player to move next, while reward
        is for the player who just moved: 1 for a win, 0 otherwise (draws included).
        Raises ValueError on an illegal action or when the game is already over."""
        if self._done:
            raise ValueError('game is over, call reset()')
        mover = self._player
        if self.move(action) == -1:
            raise ValueError('illegal action %r' % (action,))
        reward = 1.0 if self._winner == mover else 0.0
        return self.get_observation(), reward, self._done, False, self._info()

    def __call__(self, pos):
        return self.move(pos)

    def move(self, pos):
        n = self.size * self.size
        if not 0 <= pos < n:
            raise ValueError('pos must be in [0, %d), got %r' % (n, pos))
        if self._done:
            return -1
        elif self._state[pos]:
            return -1
        elif self.is_forbidden(pos):
            return -1
        else:
            self._state[pos] = self._player
            self._cache = {}
            result = self.check(pos)
            self._move_history.append(int(pos))
            self._player = self._next_player(len(self._move_history))
            self._done = result == 1
            return result

    def move_back(self):
        if len(self._move_history):
            move = self._move_history.pop(-1)
            self._state[move] = 0
            self._cache = {}
            self._player = self._next_player(len(self._move_history))
            self._winner = PLAYER_NONE
            self._done = False

    def check(self, pos):
        action_y, action_x = divmod(pos, self.size)

        state = self._board()
        player = self.get_player()

        match = max(
            check_match(state, player, action_x, action_y, sx, sy, self.win)
            for sx, sy in ((+1, 0), (0, +1), (+1, -1), (+1, +1))
        )

        if match >= self.win:
            self._winner = player
            return 1
        elif len(self._move_history) == self.size * self.size - 1:  # tie
            self._winner = PLAYER_NONE
            return 1
        else:
            return 0

    def __repr__(self):
        state = self._board()
        line = '+' + '-' * (self.size * 2 + 1) + '+\n'
        board = line
        for y in range(self.size):
            board += '|'
            for x in range(self.size):
                check = state[y, x]
                if check == 0:
                    board += ' -'
                elif check == 1:
                    board += ' O'
                else:
                    board += ' X'
            board += ' |\n'
        p = self.get_player()
        w = self.get_winner()
        m = len(self._move_history)
        board += line
        board += '| Player %d  Winner %d  Moves %3d' % (p, w, m)
        board += '  ' * (self.size - 15) + ' |\n'
        board += line
        return board

    def get_log(self):
        return {
            'moves': self.get_move_history(),
            'winner': self.get_winner(),
        }


class Omok(BoardGame):
    """Five in a row on 15x15. Renju rule forbids 3-3, 4-4 and overline for black."""
    size = SIZE
    win = WIN

    def __init__(self, rule='renju'):
        if rule not in RULES:
            raise ValueError('rule must be one of %s' % (RULES,))
        self.rule = rule
        super().__init__()

    def is_forbidden(self, pos):
        if self.rule != 'renju' or self._player != PLAYER_BLACK:
            return False
        if 'forbidden' in self._cache:
            return pos in self._cache['forbidden']
        return renju.is_forbidden(self._board(), pos)

    def get_forbidden(self):
        if self.rule != 'renju' or self._player != PLAYER_BLACK:
            return []
        if 'forbidden' not in self._cache:  # cleared whenever the board changes
            self._cache['forbidden'] = renju.forbidden_moves(self._board())
        return list(self._cache['forbidden'])

    def check(self, pos):
        result = super().check(pos)
        if (result == 0 and self.rule == 'renju' and self._player == PLAYER_WHITE
                and not renju.has_legal_move(self._board())):  # black has no move
            self._winner = PLAYER_NONE
            return 1
        return result


class Connect6(BoardGame):
    """Six in a row on 19x19. Black places 1 stone first, then each turn is 2 stones."""
    size = 19
    win = 6

    def _next_player(self, num_moves):
        return PLAYER_BLACK if (num_moves + 1) // 2 % 2 == 0 else PLAYER_WHITE

    def get_observation(self):
        """Float32 planes of shape (4, 19, 19): own stones, opponent stones,
        black to play, and a constant plane that is 1 on the last stone of a turn."""
        obs = super().get_observation()
        last = np.full((1, self.size, self.size), self.get_stones_left() == 1, np.float32)
        return np.concatenate([obs, last])
