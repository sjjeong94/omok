import numpy as np

WIN = 5
SIZE = 15
PLAYER_NONE = 0
PLAYER_BLACK = 1
PLAYER_WHITE = 2


class Omok:

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = np.zeros(SIZE * SIZE, np.uint8)
        self.player = PLAYER_BLACK
        self.winner = PLAYER_NONE
        self.move_history = []

    def __call__(self, pos):
        return self.move(pos)

    def move(self, pos):
        if self.winner:
            return -1
        elif self.state[pos]:
            return -1
        else:
            self.state[pos] = self.player
            result = self.check(pos)
            self.swap_player()
            self.move_history.append(int(pos))
            return result

    def swap_player(self):
        self.player ^= 3

    def get_state(self):
        return self.state.reshape(SIZE, SIZE)

    def check(self, pos):
        action_y, action_x = divmod(pos, SIZE)

        state = self.get_state()

        match_0 = 1
        for i in range(WIN):
            y = action_y
            x = action_x+(i+1)
            if(x >= SIZE):
                break
            check = state[y, x]
            if(check == self.player):
                match_0 += 1
            else:
                break
        for i in range(WIN):
            y = action_y
            x = action_x-(i+1)
            if(x < 0):
                break
            check = state[y, x]
            if(check == self.player):
                match_0 += 1
            else:
                break
        match_90 = 1
        for i in range(WIN):
            y = action_y+(i+1)
            x = action_x
            if(y >= SIZE):
                break
            check = state[y, x]
            if(check == self.player):
                match_90 += 1
            else:
                break
        for i in range(WIN):
            y = action_y-(i+1)
            x = action_x
            if(y < 0):
                break
            check = state[y, x]
            if(check == self.player):
                match_90 += 1
            else:
                break
        match_45 = 1
        for i in range(WIN):
            y = action_y-(i+1)
            x = action_x+(i+1)
            if((y < 0) or (x >= SIZE)):
                break
            check = state[y, x]
            if(check == self.player):
                match_45 += 1
            else:
                break
        for i in range(WIN):
            y = action_y+(i+1)
            x = action_x-(i+1)
            if((y >= SIZE) or (x < 0)):
                break
            check = state[y, x]
            if(check == self.player):
                match_45 += 1
            else:
                break
        match_135 = 1
        for i in range(WIN):
            y = action_y+(i+1)
            x = action_x+(i+1)
            if((y >= SIZE) or (x >= SIZE)):
                break
            check = state[y, x]
            if(check == self.player):
                match_135 += 1
            else:
                break
        for i in range(WIN):
            y = action_y-(i+1)
            x = action_x-(i+1)
            if((y < 0) or (x < 0)):
                break
            check = state[y, x]
            if(check == self.player):
                match_135 += 1
            else:
                break

        match = max(match_0, match_90, match_45, match_135)

        if(match > WIN):
            self.winner = self.player ^ 3
            return 1
        elif(match == WIN):
            self.winner = self.player
            return 1
        elif(len(self.move_history) == SIZE*SIZE):
            self.winner = PLAYER_NONE
            return 1
        else:
            return 0

    def show_state(self):
        state = self.get_state()
        board = '+------------------------------+\n'
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
            board += '|\n'
        board += '+------------------------------+\n'
        print(board)

    def get_log(self):
        return {
            'moves': self.move_history,
            'winner': self.winner,
        }
