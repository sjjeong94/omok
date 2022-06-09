import omok


cases = [
    {"moves": [112, 113, 127, 128, 142, 143, 157, 158, 172], "winner":1},
    {"moves": [112, 97, 111, 96, 110, 95, 109, 94, 108], "winner":1},
    {"moves": [112, 127, 126, 141, 140, 155, 154, 169, 168], "winner":1},
    {"moves": [112, 111, 96, 95, 80, 79, 64, 63, 48], "winner":1},
    {"moves": [112, 127, 126, 142, 128, 157, 144, 172, 160, 187], "winner":2},
    {"moves": [112, 111, 96, 110, 95, 109, 94, 108, 93, 107], "winner":2},
    {"moves": [112, 111, 127, 95, 142, 79, 113, 63, 97, 47], "winner":2},
    {"moves": [112, 128, 98, 114, 84, 100, 70, 86, 42, 72], "winner":2},
    {"moves": [112, 113, 127, 128, 142, 143,
               157, 158, 187, 188, 172], "winner":2},
    {"moves": [112, 111, 97, 96, 82, 81, 52, 51, 37, 36, 53, 66], "winner":1}
]


def test_cases():
    env = omok.Omok()
    for case in cases:
        env.reset()
        for move in case['moves']:
            env(move)
        assert env.get_winner() == case['winner'], print("Fail -> ", case)
