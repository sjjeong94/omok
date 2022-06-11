import omok


def play():
    agent = omok.OmokAgent()
    game = omok.OmokGame()
    while game():
        state = game.env.get_state()
        player = game.env.get_player()
        if player == 2:
            action = agent(state, player)
            result = game.env(action)


if __name__ == '__main__':
    play()
