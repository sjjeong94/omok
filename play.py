import omok


def play():
    agent = omok.OmokAgent(model_index=0)
    game = omok.OmokGame(agent=agent)
    while game():
        pass


if __name__ == '__main__':
    play()
