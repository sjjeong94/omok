import os
import time
import json
import threading
import webbrowser
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from omok import Omok, Connect6
from omok.version import VERSION


class OmokGame:
    def __init__(self, agent=None, rule='renju', env=None):
        self.env = Omok(rule=rule) if env is None else env
        self.agent = agent
        self.lock = threading.Lock()

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
            'version': VERSION,
        }
        if probs and self.agent is not None and not self.env.is_done():
            status['probs'] = self.agent.get_probs(
                self.env.get_state(), self.env.get_player()).round(4).tolist()
        if analysis and self.agent is not None and not self.env.is_done():
            status['analysis'] = self.get_analysis()
        return status

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
