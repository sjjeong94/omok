import omok


def play():
    agent = omok.OmokAgent(model_index=1)
    game = omok.OmokGame(agent=agent)
    while game():
        pass


play()
