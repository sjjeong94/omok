import numpy as np
from omok import renju

WIN = 5
SIZE = 15
PLAYER_NONE = 0
PLAYER_BLACK = 1
PLAYER_WHITE = 2
RULES = ('renju', 'freestyle')


def check_match(state, player, action_x, action_y, sx, sy):
    match = 1
    for i in range(WIN):
        y = action_y+(i+1)*sy
        x = action_x+(i+1)*sx
        if (x < 0) or (x >= SIZE) or (y < 0) or (y >= SIZE):
            break
        check = state[y, x]
        if (check == player):
            match += 1
        else:
            break
    for i in range(WIN):
        y = action_y-(i+1)*sy
        x = action_x-(i+1)*sx
        if (x < 0) or (x >= SIZE) or (y < 0) or (y >= SIZE):
            break
        check = state[y, x]
        if (check == player):
            match += 1
        else:
            break
    return match


class Omok:
    def __init__(self, rule='renju'):
        if rule not in RULES:
            raise ValueError('rule must be one of %s' % (RULES,))
        self.rule = rule
        self.reset()

    def reset(self):
        self.__state = np.zeros(SIZE * SIZE, np.uint8)
        self.__player = PLAYER_BLACK
        self.__winner = PLAYER_NONE
        self.__done = False
        self.__move_history = []

    def __board(self):
        return self.__state.reshape(SIZE, SIZE)

    def get_state(self):
        return self.__board().copy()

    def get_player(self):
        return self.__player

    def get_winner(self):
        return self.__winner

    def is_done(self):
        return self.__done

    def get_move_history(self):
        return list(self.__move_history)

    def is_forbidden(self, pos):
        if self.rule != 'renju' or self.__player != PLAYER_BLACK:
            return False
        return renju.is_forbidden(self.__board(), pos)

    def get_forbidden(self):
        if self.rule != 'renju' or self.__player != PLAYER_BLACK:
            return []
        return renju.forbidden_moves(self.__board())

    def get_legal_mask(self):
        """Boolean mask of shape (225,) for the moves the current player can make."""
        if self.__done:
            return np.zeros(SIZE * SIZE, bool)
        mask = self.__state == PLAYER_NONE
        mask[self.get_forbidden()] = False
        return mask

    def get_observation(self):
        """Float32 planes of shape (3, 15, 15) from the current player's view:
        own stones, opponent stones, and a constant plane that is 1 if black to play."""
        board = self.__board()
        player = self.__player
        return np.stack([
            board == player,
            board == player ^ 3,
            np.full((SIZE, SIZE), player == PLAYER_BLACK),
        ]).astype(np.float32)

    def __call__(self, pos):
        return self.move(pos)

    def move(self, pos):
        if not 0 <= pos < SIZE * SIZE:
            raise ValueError('pos must be in [0, %d), got %r' % (SIZE * SIZE, pos))
        if self.__done:
            return -1
        elif self.__state[pos]:
            return -1
        elif self.is_forbidden(pos):
            return -1
        else:
            self.__state[pos] = self.__player
            result = self.check(pos)
            self.__swap_player()
            self.__move_history.append(int(pos))
            self.__done = result == 1
            return result

    def move_back(self):
        if len(self.__move_history):
            move = self.__move_history.pop(-1)
            self.__state[move] = 0
            self.__swap_player()
            self.__winner = PLAYER_NONE
            self.__done = False

    def __swap_player(self):
        self.__player ^= 3

    def check(self, pos):
        action_y, action_x = divmod(pos, SIZE)

        state = self.__board()
        player = self.get_player()

        match_0 = check_match(state, player, action_x, action_y, +1, 0)
        match_90 = check_match(state, player, action_x, action_y, 0, +1)
        match_45 = check_match(state, player, action_x, action_y, +1, -1)
        match_135 = check_match(state, player, action_x, action_y, +1, +1)

        check_result = [match_0, match_45, match_90, match_135]

        match = max(check_result)

        if match >= WIN:
            self.__winner = player
            return 1
        elif len(self.__move_history) == SIZE*SIZE - 1:  # tie
            self.__winner = PLAYER_NONE
            return 1
        elif (self.rule == 'renju' and player == PLAYER_WHITE
              and not renju.has_legal_move(state)):  # black has no move
            self.__winner = PLAYER_NONE
            return 1
        else:
            return 0
    
    def __repr__(self):
        state = self.__board()
        board = '+-------------------------------+\n'
        for y in range(SIZE):
            board += '|'
            for x in range(SIZE):
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
        m = len(self.get_move_history())
        board += '+-------------------------------+\n'
        board += '| Player %d  Winner %d  Moves %3d |\n' % (p, w, m)
        board += '+-------------------------------+\n'
        return board

    def get_log(self):
        return {
            'moves': self.get_move_history(),
            'winner': self.get_winner(),
        }
