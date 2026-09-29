import omok


def test_agent():
    env = omok.Omok()
    agent = omok.OmokAgent(model_index=1)

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


def test_get_probs():
    env = omok.Omok()
    agent = omok.OmokAgent(model_index=1)
    for move in [112, 111, 96, 97, 128]:
        env(move)
    state = env.get_state()
    player = env.get_player()

    probs = agent.get_probs(state, player)
    assert probs.shape == (225,)
    assert abs(probs.sum() - 1) < 1e-5
    assert (probs[state.reshape(-1) != 0] == 0).all()

    # averaged over symmetries, so transforming the board transforms the probs
    for code in range(8):
        state_t, _ = agent.transform(state, 0, code)
        probs_t = agent.get_probs(state_t, player).reshape(15, 15)
        expected, _ = agent.transform(probs.reshape(15, 15), 0, code)
        assert abs(probs_t - expected).max() < 1e-5
