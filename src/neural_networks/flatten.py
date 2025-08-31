from layer_base import Layer
import numpy as np

class Flatten(Layer):
    def __init__(self, keep_batch: bool = True):
        super().__init__()
        self.keep_batch = keep_batch
        self.original_shape = None
        self._added_fake_batch = False  # only for truly 1D inputs.

    def forward(self, inputs):
        x = np.asarray(inputs)
        self.original_shape = x.shape
        self._added_fake_batch = False

        # Only (B,C,H,W) is treated as batched.
        if x.ndim == 4:
            B = x.shape[0]
            return x.reshape(B, -1)

        # Otherwise: flatten everything; optionally add a fake batch.
        flat = x.reshape(-1)
        if self.keep_batch:
            # Track if we faked batch for pure 1D (helps backward niceties).
            self._added_fake_batch = (x.ndim == 1)
            return flat[None, :]
        return flat

    def backward(self, grads):
        g = np.asarray(grads)

        if self.original_shape is None:
            raise RuntimeError("Flatten.backward called before forward.")

        # If we flattened a (B,C,H,W), expect grads as (B,F); reshape back.
        if len(self.original_shape) == 4:
            if g.ndim == 1:  # tolerate single-sample grads.
                g = g[None, :]
            return g.reshape(self.original_shape)

        # For non-4D originals, we flattened fully.
        if self.keep_batch and self._added_fake_batch and g.ndim == 2 and g.shape[0] == 1:
            g = g[0]
        return g.reshape(self.original_shape)

    def has_params(self):
        return False

    def describe(self):
        return "Flatten"
