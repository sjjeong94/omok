import numpy as np

WIN = 5
SIZE = 15
PLAYER_NONE = 0
PLAYER_BLACK = 1
PLAYER_WHITE = 2


def check_match(state, player, action_x, action_y, sx, sy):
    match = 1
    for i in range(WIN):
        y = action_y+(i+1)*sy
        x = action_x+(i+1)*sx
        if (x < 0) or (x >= SIZE) or (y < 0) or (y >= SIZE):
            break
        check = state[y, x]
        if(check == player):
            match += 1
        else:
            break
    for i in range(WIN):
        y = action_y-(i+1)*sy
        x = action_x-(i+1)*sx
        if (x < 0) or (x >= SIZE) or (y < 0) or (y >= SIZE):
            break
        check = state[y, x]
        if(check == player):
            match += 1
        else:
            break
    return match


class Omok:
    def __init__(self):
        self.reset()

    def reset(self):
        self.__state = np.zeros(SIZE * SIZE, np.uint8)
        self.__player = PLAYER_BLACK
        self.__winner = PLAYER_NONE
        self.__move_history = []

    def get_state(self):
        return self.__state.reshape(SIZE, SIZE)

    def get_player(self):
        return self.__player

    def get_winner(self):
        return self.__winner

    def get_move_history(self):
        return self.__move_history

    def __call__(self, pos):
        return self.move(pos)

    def move(self, pos):
        if self.__winner:
            return -1
        elif self.__state[pos]:
            return -1
        else:
            self.__state[pos] = self.__player
            result = self.check(pos)
            self.__swap_player()
            self.__move_history.append(int(pos))
            return result

    def move_back(self):
        if len(self.__move_history):
            move = self.__move_history.pop(-1)
            self.__state[move] = 0
            self.__swap_player()
            self.__winner = 0

    def __swap_player(self):
        self.__player ^= 3

    def check(self, pos):
        action_y, action_x = divmod(pos, SIZE)

        state = self.get_state()
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
        else:
            return 0

    def show_state(self):
        state = self.get_state()
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
        print(board)

    def get_log(self):
        return {
            'moves': self.get_move_history(),
            'winner': self.get_winner(),
        }
