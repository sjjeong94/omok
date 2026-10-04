import argparse
import omok


def play():
    parser = argparse.ArgumentParser(prog='python -m omok')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--game', choices=('omok', 'connect6'), default='omok')
    parser.add_argument('--rule', choices=omok.env.RULES, default='renju')
    parser.add_argument('--agent', choices=('alphazero', 'policy'), default='alphazero',
                        help='alphazero: AlphaZeroAgent (default); policy: OmokAgent (a.onnx / b.onnx)')
    parser.add_argument('--model-index', type=int, default=1, help='OmokAgent model (policy agent)')
    parser.add_argument('--simulations', type=int, default=200,
                        help='AlphaZeroAgent search simulations per move (0: network only)')
    parser.add_argument('--no-agent', action='store_true')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()

    if args.game == 'connect6':
        # no trained model for connect6 yet
        game = omok.OmokGame(env=omok.Connect6())
    else:
        if args.no_agent:
            agent = None
        elif args.agent == 'alphazero':
            agent = omok.AlphaZeroAgent(rule=args.rule, simulations=args.simulations)
        else:
            agent = omok.OmokAgent(model_index=args.model_index, rule=args.rule)
        game = omok.OmokGame(agent=agent, rule=args.rule)
    game.run(args.host, args.port, open_browser=not args.no_browser)


play()
