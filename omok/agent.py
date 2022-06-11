import numpy as np
import onnxruntime


class OmokAgent:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path)

    def __call__(self, state, player):
        if np.random.rand() < 0.5:
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
        action = np.argmax(outs[0].squeeze())
        return action
