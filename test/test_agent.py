import omok


def test_agent():
    env = omok.Omok()
    agent = omok.OmokAgent()

    num_games = 100
    for _ in range(num_games):
        env.reset()
        while True:
            state = env.get_state()
            player = env.get_player()
            action = agent(state, player)
            result = env(action)
            if result:
                break
        assert result == 1
