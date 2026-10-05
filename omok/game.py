import os
import time
import json
import threading
import webbrowser
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from omok import Omok, Connect6
from omok.alphazero import position_key
from omok.env import PLAYER_BLACK, PLAYER_WHITE
from omok.version import VERSION


class SearchJob:
    """A search of one position running in the background, keeping a frame of its progress per batch."""

    def __init__(self, agent, env):
        self.moves = env.get_move_history()
        self.frames = []
        self.stopped = threading.Event()
        state, player = env.get_state(), env.get_player()
        self.thread = threading.Thread(target=self.run, args=(agent, state, player), daemon=True)
        self.thread.start()

    def run(self, agent, state, player):
        agent.watch(state, player, self.on_frame)

    def on_frame(self, frame):
        self.frames.append(frame)
        return not self.stopped.is_set()

    @property
    def done(self):
        return not self.thread.is_alive()

    def stop(self):
        self.stopped.set()
        self.thread.join()


class OmokGame:
    def __init__(self, agent=None, rule='renju', env=None):
        self.env = Omok(rule=rule) if env is None else env
        self.agent = agent
        self.lock = threading.Lock()
        self.job = None  # the live search being watched, if any
        self.values = {}  # position key -> the network's value, for the player to move

    def can_watch(self):
        return getattr(self.agent, 'simulations', 0) > 0 and hasattr(self.agent, 'watch')

    def watch_search(self, since=0, final=False):
        """Start (or continue) a live search of the current position; returns its frames from `since` on
        (or, with `final`, only the last one, once the search is done)."""
        if not self.can_watch():
            return None
        moves = self.env.get_move_history()
        if self.env.is_done():
            return {'moves': len(moves), 'frames': [], 'done': True, 'total': 0}
        if self.job is None or self.job.moves != moves:
            self.stop_search()
            self.job = SearchJob(self.agent, self.env)
        job = self.job
        done = job.done  # read before the frames, so a finished job's frames are all there
        frames = (job.frames[-1:] if done else []) if final else job.frames[since:]
        return {'moves': len(moves), 'frames': frames, 'done': done, 'total': self.agent.simulations}

    def stop_search(self):
        if self.job is not None:
            self.job.stop()
            self.job = None

    def finish_search(self):
        """Let the live search finish, so the agent uses it (it is kept in the agent's cache)."""
        if self.job is not None:
            self.job.thread.join()

    def get_status(self, probs=False, analysis=False):
        status = {
            'state': self.env.get_state().reshape(-1).tolist(),
            'player': self.env.get_player(),
            'winner': self.env.get_winner(),
            'done': self.env.is_done(),
            'moves': self.env.get_move_history(),
            'forbidden': self.env.get_forbidden(),
            'stones_left': self.env.get_stones_left(),
            'size': self.env.size,
            'game': 'connect6' if isinstance(self.env, Connect6) else 'omok',
            'rule': getattr(self.env, 'rule', None),
            'agent': self.agent is not None,
            'watch': self.can_watch(),
            'graph': hasattr(self.agent, 'get_values'),
            'version': VERSION,
        }
        if probs and self.agent is not None and not self.env.is_done():
            status['probs'] = self.agent.get_probs(
                self.env.get_state(), self.env.get_player()).round(4).tolist()
        if analysis and self.agent is not None and not self.env.is_done():
            status['analysis'] = self.get_analysis()
        return status

    def get_win_rates(self, budget=None):
        """Black's winning chances in each position of the game (index i: after i moves), in [0, 1]:
        'network', the network's estimate (or the result, once the game is over), and 'search', the value of the
        agent's search of the position (None where it did not search).

        The network evaluates at most `budget` new positions, from the start of the game on; the rest are None
        for now, and 'pending' counts them."""
        env, positions = self.env.clone(), []
        while True:
            positions.append((env.get_state().copy(), env.get_player(), env.is_done(), env.get_winner()))
            if not env.get_move_history():
                break
            env.move_back()
        positions.reverse()
        missing = [(state, player) for state, player, done, _ in positions
                   if not done and position_key(state, player) not in self.values]
        new = missing[:budget]
        self.values.update(zip((position_key(*p) for p in new), self.agent.get_values(new)))

        def black(value, player):  # a value for the player to move, as Black's winning chance
            return round((1 + value) / 2 if player == PLAYER_BLACK else (1 - value) / 2, 4)

        network, search = [], []
        for state, player, done, winner in positions:
            if done:
                network.append({PLAYER_BLACK: 1.0, PLAYER_WHITE: 0.0}.get(winner, 0.5))
                search.append(None)
                continue
            value = self.values.get(position_key(state, player))
            network.append(None if value is None else black(value, player))
            value = self.agent.get_search_value(state, player) if hasattr(self.agent, 'get_search_value') else None
            search.append(None if value is None else black(value, player))
        return {'network': network, 'search': search, 'pending': len(missing) - len(new)}

    def get_analysis(self):
        state, player = self.env.get_state(), self.env.get_player()
        if hasattr(self.agent, 'get_analysis'):
            return self.agent.get_analysis(state, player)
        prior = self.agent.get_probs(state, player)  # a policy-only agent: just its probabilities
        top = np.argsort(-prior, kind='stable')[:10]
        return {'simulations': 0, 'prior': prior.round(4).tolist(), 'value': None,
                'lines': [{'pos': int(m), 'prior': round(float(prior[m]), 4), 'pv': [int(m)]}
                          for m in top if prior[m] > 0]}

    def move(self, pos):
        self.env.move(int(pos))

    def agent_move(self):
        if self.agent is not None and not self.env.is_done():
            state = self.env.get_state()
            player = self.env.get_player()
            self.env(self.agent(state, player))

    def save_log(self):
        os.makedirs('logs', exist_ok=True)
        file_name = 'logs/%d.json' % int(time.time())
        with open(file_name, 'w') as f:
            json.dump(self.env.get_log(), f, separators=(',', ':'))
        return file_name

    def handle(self, action, body):
        with self.lock:
            if action == 'search':
                return self.watch_search(int(body.get('since', 0)), bool(body.get('final')))
            if action == 'graph':  # a few positions per request, so that moves are not held up for long
                return self.get_win_rates(budget=8) if hasattr(self.agent, 'get_values') else None
            if action in ('move', 'back', 'reset'):
                self.stop_search()
            elif action == 'agent' or body.get('analysis') or body.get('probs'):
                self.finish_search()  # one search at a time: the agent is not thread-safe
            if action == 'state':
                pass
            elif action == 'move':
                self.move(body['pos'])
                if body.get('reply'):
                    self.agent_move()
            elif action == 'agent':
                self.agent_move()
            elif action == 'back':
                self.env.move_back()
            elif action == 'reset':
                self.env.reset()
            elif action == 'save':
                return {'file': self.save_log()}
            else:
                return None
            return self.get_status(probs=body.get('probs'), analysis=body.get('analysis'))

    def make_handler(self):
        game = self
        page = resources.files('omok').joinpath('static/index.html').read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def send(self, code, data, content_type):
                self.send_response(code)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def send_json(self, obj):
                if obj is None:
                    self.send(404, b'{}', 'application/json')
                else:
                    self.send(200, json.dumps(obj).encode(), 'application/json')

            def do_GET(self):
                if self.path in ('/', '/index.html'):
                    self.send(200, page, 'text/html; charset=utf-8')
                elif self.path == '/api/state':
                    self.send_json(game.handle('state', {}))
                else:
                    self.send(404, b'Not Found', 'text/plain')

            def do_POST(self):
                if not self.path.startswith('/api/'):
                    return self.send(404, b'Not Found', 'text/plain')
                length = int(self.headers.get('Content-Length') or 0)
                try:
                    body = json.loads(self.rfile.read(length) or b'{}')
                    result = game.handle(self.path[len('/api/'):], body)
                except (ValueError, KeyError, TypeError):
                    return self.send(400, b'{}', 'application/json')
                self.send_json(result)

            def log_message(self, format, *args):
                pass

        return Handler

    def run(self, host='127.0.0.1', port=8000, open_browser=True):
        server = ThreadingHTTPServer((host, port), self.make_handler())
        url = 'http://%s:%d' % (host, server.server_address[1])
        print('Omok %s running at %s (Ctrl+C to quit)' % (VERSION, url))
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
