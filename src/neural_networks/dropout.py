import numpy as np
from layer_base import Layer

class DropoutLayer(Layer):
    def __init__(self, drop_prob=0.5):
        if not (0 <= drop_prob < 1):
            raise ValueError("drop_prob must be in [0,1)")
        self.drop_prob = float(drop_prob)
        self.mask = None
        self.training = True
        self._scale = 1.0 / (1.0 - self.drop_prob) if self.drop_prob < 1.0 else np.inf

    def forward(self, inputs):
        x = np.asarray(inputs)
        if self.training:
            # Bernoulli(keep_prob) mask, same shape as x.
            self.mask = np.random.binomial(1, 1.0 - self.drop_prob, size=x.shape).astype(x.dtype)
            return x * self.mask * self._scale
        return x

    def backward(self, grads):
        g = np.asarray(grads)
        if self.training and self.mask is not None:
            return g * self.mask * self._scale
        return g

    def params(self):
        return {"w": [], "b": []}

    def apply_grads(self, grads, lr, momentum, clip, weight_decay, state, layer_key):
        return  # no params

    def has_params(self):
        return False

    def describe(self):
        return f"Dropout: p={self.drop_prob}"
