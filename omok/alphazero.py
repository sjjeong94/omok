"""AlphaZero agent: a policy-value network trained by self-play, with an optional PUCT search.

The networks were trained in https://github.com/sjjeong94/omok-rl (Stage 6: Gumbel AlphaZero), one per rule.
Over 100 games from random 4-move openings against `OmokAgent(model_index=1)` (b.onnx) they score
0.76 (Renju) and 0.82 (freestyle) with the network alone, and 0.91 and 0.88 with a 200-simulation search.

The interface is the same as `OmokAgent`: `agent(state, player)` returns a move and `agent.get_probs(state, player)`
the move probabilities, so it also works in the web UI (`python -m omok --agent alphazero`).
"""

import numpy as np
import onnxruntime

from omok import transforms
from omok.agent import check_model
from omok.env import PLAYER_NONE, Omok

MODELS = {'renju': 'alphazero-renju.onnx', 'freestyle': 'alphazero-freestyle.onnx'}


class Node:
    """A position in the search tree, with the statistics of its legal moves."""

    __slots__ = ('moves', 'prior', 'n', 'w', 'children', 'visits')

    def __init__(self, moves, prior):
        self.moves = moves  # legal moves
        self.prior = prior  # the network's probability of each move
        self.n = np.zeros(len(moves))  # visits of each move
        self.w = np.zeros(len(moves))  # total value of each move, for the player who makes it
        self.children = [None] * len(moves)
        self.visits = 0


def env_from_state(state, player, rule='renju'):
    """An `Omok` env with the stones of `state` (15x15, 0 / 1 black / 2 white) and `player` to move.

    The order of the moves is unknown, so the move history only has the right length (it decides whose turn
    follows each move); its contents are placeholders and the last-move observation plane is meaningless."""
    board = np.asarray(state, np.uint8).reshape(-1)
    env = Omok(rule=rule)
    stones = np.flatnonzero(board).tolist()
    if env._next_player(len(stones)) != player:
        raise ValueError('player %d cannot be to move with %d black and %d white stones'
                         % (player, (board == 1).sum(), (board == 2).sum()))
    env._state = board.copy()
    env._player = player
    env._move_history = stones
    env._cache = {}
    return env


class AlphaZeroAgent:
    """Plays with an AlphaZero network, alone (`simulations=0`) or with a PUCT search of `simulations` simulations.

    - `rule`: 'renju' or 'freestyle'; picks the network (downloaded to ./omok_assets on first use) and the rules
      the search plays by.
    - `random_transform`: show each position to the network through a random one of the 8 board symmetries, as in
      training (and as `OmokAgent` does), so that games vary.
    - `leaves`: positions evaluated per network call during the search (virtual loss), which makes it faster.

    A search of 200 simulations takes about 1 s per move on a CPU.
    """

    def __init__(self, rule='renju', simulations=0, model_path=None, random_transform=True, c_puct=1.5, leaves=8,
                 seed=None):
        if rule not in MODELS:
            raise ValueError('rule must be one of %s' % (tuple(MODELS),))
        self.rule, self.simulations, self.c_puct, self.leaves = rule, simulations, c_puct, leaves
        self.session = onnxruntime.InferenceSession(model_path or check_model(MODELS[rule]),
                                                    providers=['CPUExecutionProvider'])
        self.random_transform = random_transform
        self.rng = np.random.default_rng(seed)
        perms = np.stack([transforms.transform_board(np.arange(225).reshape(15, 15), code).ravel()
                          for code in range(8)])
        self.perms, self.inverse = perms, np.argsort(perms, axis=1)
        self._cache = (None, None)  # (position key, root of its search)

    # ------------------------------------------------------------------ network

    def evaluate(self, obs, masks, codes=None):
        """obs (B, 5, 15, 15), masks (B, 225) -> legal-move probabilities (B, 225) and values (B,) for the player
        to move. Each position is shown through the symmetry `codes[i]` (random by default) and mapped back."""
        n = len(obs)
        if codes is None:
            codes = self.rng.integers(8, size=n) if self.random_transform else np.zeros(n, np.int64)
        x = np.take_along_axis(obs.reshape(n, 5, 225), self.perms[codes][:, None], 2).reshape(n, 5, 15, 15)
        logits, values = self.session.run(None, {'obs': x.astype(np.float32)})
        logits = np.take_along_axis(logits, self.inverse[codes], 1)
        logits = np.where(masks, logits, -np.inf)
        probs = np.exp(logits - logits.max(1, keepdims=True))
        return probs / probs.sum(1, keepdims=True), values

    def network_probs(self, env):
        """The network's move probabilities, averaged over the 8 board symmetries."""
        obs = np.repeat(env.get_observation()[None], 8, 0)
        mask = np.repeat(env.get_legal_mask()[None], 8, 0)
        return self.evaluate(obs, mask, np.arange(8))[0].mean(0)

    # ------------------------------------------------------------------ search

    def select(self, node):
        q = np.divide(node.w, node.n, out=np.zeros_like(node.w), where=node.n > 0)
        u = self.c_puct * node.prior * np.sqrt(node.visits + 1) / (1 + node.n)
        return int(np.argmax(q + u))

    def search(self, env):
        """Run a PUCT search from `env` (left unchanged); returns the root."""
        mask = env.get_legal_mask()
        probs, _ = self.evaluate(env.get_observation()[None], mask[None])
        moves = np.flatnonzero(mask)
        root = Node(moves, probs[0][moves])
        env = env.clone()
        while root.visits < self.simulations:
            pending, taken = [], set()
            for _ in range(min(self.leaves, self.simulations - root.visits)):
                node, path = root, []
                while True:
                    i = self.select(node)
                    path.append((node, i))
                    env.move(int(node.moves[i]))
                    if env.is_done() or node.children[i] is None:
                        break
                    node = node.children[i]
                if env.is_done():  # exact result: a win for the player who just moved, or a draw
                    backup(path, 0.0 if env.get_winner() == PLAYER_NONE else 1.0)
                elif (id(node), i) in taken:  # the same leaf twice: evaluate what we have
                    for _ in path:
                        env.move_back()
                    break
                else:
                    taken.add((id(node), i))
                    pending.append((path, env.get_observation(), env.get_legal_mask()))
                    for nd, j in path:  # virtual loss: the next walk prefers another path
                        nd.n[j] += 1
                        nd.w[j] -= 1
                for _ in path:
                    env.move_back()
            if not pending:
                continue
            probs, values = self.evaluate(np.stack([p[1] for p in pending]), np.stack([p[2] for p in pending]))
            for (path, _, mask), p, v in zip(pending, probs, values):
                for nd, j in path:
                    nd.n[j] -= 1
                    nd.w[j] += 1
                node, i = path[-1]
                moves = np.flatnonzero(mask)
                node.children[i] = Node(moves, p[moves])
                backup(path, -float(v))  # v is for the player to move at the leaf, the opponent of the mover
        return root

    def root(self, state, player):
        key = (np.asarray(state, np.uint8).tobytes(), int(player))
        if self._cache[0] != key:
            self._cache = key, self.search(env_from_state(state, player, self.rule))
        return self._cache[1]

    # ------------------------------------------------------------------ OmokAgent interface

    def __call__(self, state, player):
        if self.simulations <= 0:
            env = env_from_state(state, player, self.rule)
            probs, _ = self.evaluate(env.get_observation()[None], env.get_legal_mask()[None])
            return int(np.argmax(probs[0]))
        root = self.root(state, player)
        best = np.flatnonzero(root.n == root.n.max())
        return int(root.moves[self.rng.choice(best)])

    def get_probs(self, state, player):
        """Move probabilities (225,): the network's (averaged over the 8 symmetries), or the search's visit shares."""
        if self.simulations <= 0:
            return self.network_probs(env_from_state(state, player, self.rule)).astype(np.float32)
        root = self.root(state, player)
        probs = np.zeros(225, np.float32)
        probs[root.moves] = root.n / root.n.sum()
        return probs


def backup(path, value):
    """Add `value` (for the player who made the last move of `path`) to every move on the path."""
    for node, i in reversed(path):
        node.n[i] += 1
        node.w[i] += value
        node.visits += 1
        value = -value
