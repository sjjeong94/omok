import os
import numpy as np
import onnxruntime
from urllib import request

import omok.transforms

models_link = 'https://github.com/sjjeong94/omok/raw/main/models/'
models_path = './omok_assets'
models_name = [
    'a.onnx',
    'b.onnx',
]


def check_models(model_index: int = 0):
    if not os.path.exists(models_path):
        os.makedirs(models_path, exist_ok=True)
    name = models_name[model_index]
    path = os.path.join(models_path, name)
    if not os.path.exists(path):
        link = models_link + name
        request.urlretrieve(link, path)
    return path


def softmax(z):
    z = np.exp(z)
    return z / np.sum(z)


class Transform:
    def __init__(self):
        self.hflip = omok.transforms.HorizontalFlip()
        self.vflip = omok.transforms.VerticalFlip()
        self.trans = omok.transforms.Transpose()

    def __call__(self, state, action, code=0):
        h = (code >> 0) & 1
        v = (code >> 1) & 1
        t = (code >> 2) & 1
        if h:
            state, action = self.hflip(state, action)
        if v:
            state, action = self.vflip(state, action)
        if t:
            state, action = self.trans(state, action)
        return state, action

    def invert(self, state, action, code=0):
        h = (code >> 0) & 1
        v = (code >> 1) & 1
        t = (code >> 2) & 1
        if t:
            state, action = self.trans(state, action)
        if v:
            state, action = self.vflip(state, action)
        if h:
            state, action = self.hflip(state, action)
        return state, action


class OmokAgent:
    def __init__(
        self,
        model_path: str = None,
        model_index: int = 0,
        random_action=True,
    ):
        if model_path is None:
            model_path = check_models(model_index)
        self.session = onnxruntime.InferenceSession(model_path)
        self.random_action = random_action
        self.transform = Transform()

    def __call__(self, state, player):
        if self.random_action:
            # Random Action for Diversity
            code = np.random.randint(0, 8)
            state_t, _ = self.transform(state, 0, code)
            action_t = self.inference(state_t, player)
            state, action = self.transform.invert(state_t, action_t, code)
        else:
            action = self.inference(state, player)
        return action

    def inference(self, state, player):
        opponent = player ^ 3
        board = np.int8(state == player) - np.int8(state == opponent)
        x = board.astype(np.float32)
        x = np.reshape(x, (1, 1, 15, 15))
        outs = self.session.run(None, {'input': x})
        out = softmax(outs[0].squeeze())
        out[(state.reshape(-1) != 0)] = -1  # masking
        action = np.argmax(out)
        return action
