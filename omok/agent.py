import os
from random import random
import numpy as np
import onnxruntime
from urllib import request

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


class OmokAgent:
    def __init__(
        self,
        model_path: str = None,
        model_index: int = 0,
        random_transpose=True,
    ):
        if model_path is None:
            model_path = check_models(model_index)
        self.session = onnxruntime.InferenceSession(model_path)
        self.random_transpose = random_transpose

    def __call__(self, state, player):
        if self.random_transpose and (np.random.rand() < 0.5):
            # Random Transpose for Diversity
            state_t = state.T
            action_t = self.inference(state_t, player)
            y, x = divmod(action_t, 15)
            action = x * 15 + y
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
