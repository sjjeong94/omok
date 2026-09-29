import argparse
import omok


def play():
    parser = argparse.ArgumentParser(prog='python -m omok')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model-index', type=int, default=1)
    parser.add_argument('--no-agent', action='store_true')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()

    agent = None if args.no_agent else omok.OmokAgent(model_index=args.model_index)
    game = omok.OmokGame(agent=agent)
    game.run(args.host, args.port, open_browser=not args.no_browser)


play()
