import json
import omok


def test_cases():
    with open('./test/cases.json', 'rb') as f:
        cases = json.load(f)

    env = omok.Omok()
    for case in cases:
        env.reset()
        for move in case['moves']:
            result = env(move)
            assert result != -1, print('Fail -> ', case)
        assert env.get_winner() == case['winner'], print('Fail -> ', case)
