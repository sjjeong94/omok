import omok


def play():
    game = omok.OmokGame(agent=omok.OmokAgent())
    while game():
        pass


if __name__ == '__main__':
    play()
