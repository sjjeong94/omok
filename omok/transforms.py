import numpy as np


class HorizontalFlip:
    def __call__(self, state, action):
        state = np.fliplr(state)
        y, x = divmod(action, 15)
        action = y * 15 + (14 - x)
        return state, action


class VerticalFlip:
    def __call__(self, state, action):
        state = np.flipud(state)
        y, x = divmod(action, 15)
        action = (14 - y) * 15 + x
        return state, action


class Transpose:
    def __call__(self, state, action):
        state = state.T
        y, x = divmod(action, 15)
        action = x * 15 + y
        return state, action
