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

    def __call__(self, state, action, code=0, invert=False):
        compose = []
        if (code >> 0) & 1:
            compose.append(self.hflip)
        if (code >> 1) & 1:
            compose.append(self.vflip)
        if (code >> 2) & 1:
            compose.append(self.trans)

        if invert:
            compose = compose[::-1]

        for func in compose:
            state, action = func(state, action)

        return state, action


class OmokAgent:
    def __init__(
        self,
        model_path: str = None,
        model_index: int = 0,
        random_transform=True,
        sampling=False,
    ):
        if model_path is None:
            model_path = check_models(model_index)
        self.session = onnxruntime.InferenceSession(model_path)
        self.transform = Transform()
        self.random_transform = random_transform
        self.sampling = sampling

    def __call__(self, state, player):
        if self.random_transform:
            # Random Action for Diversity
            code = np.random.randint(0, 8)
            state_t, _ = self.transform(state, 0, code)
            action_t = self.inference(state_t, player)
            state, action = self.transform(state_t, action_t, code, True)
        else:
            action = self.inference(state, player)
        return action

    @staticmethod
    def sample(prob):
        cumsum = np.cumsum(prob)
        sample = np.random.uniform(0, cumsum[-1])
        for i, c in enumerate(cumsum):
            if c > sample:
                break
        return i

    def inference(self, state, player):
        opponent = player ^ 3
        board = np.int8(state == player) - np.int8(state == opponent)
        x = board.astype(np.float32)
        x = np.reshape(x, (1, 1, 15, 15))
        outs = self.session.run(None, {'input': x})
        out = softmax(outs[0].squeeze())
        out[(state.reshape(-1) != 0)] = 0  # masking

        if self.sampling:
            action = self.sample(out)
        else:
            action = np.argmax(out)

        return action
