import random
from layer_base import Layer

class DropoutLayer(Layer):
    def __init__(self, drop_prob=0.5):
        if not (0 <= drop_prob < 1):
            raise ValueError("drop_prob must be in [0,1)")
        self.drop_prob = drop_prob
        self.mask = None
        self.training = True

    def forward(self, inputs):
        if self.training:
            self.mask = [0 if random.random() < self.drop_prob else 1 for _ in inputs]
            scale = 1.0 / (1.0 - self.drop_prob)
            return [x * m * scale for x, m in zip(inputs, self.mask)]
        return inputs

    def backward(self, grads):
        if self.training and self.mask is not None:
            return [g * m for g, m in zip(grads, self.mask)]
        return grads

    def params(self):
        """
        Return empty params since Dropout has no weights.
        """
        return {"w": [], "b": []}

    def apply_grads(self, grads, lr, momentum, clip, weight_decay, state, layer_key):
        """
        Dropout has no parameters, so nothing to update.
        """
        return

    def has_params(self):
        return False

    def describe(self):
        return f"Dropout: p={self.drop_prob}"