from layer_base import Layer
import numpy as np
from utils import init_weights, init_bias
from activations import get_activation

class Dense(Layer):
    """
    Fully connected layer.

    Shapes:
      Forward: inputs (B, in) -> outputs (B, out).
      Backward: grads (B, out) -> input_grads (B, in)

    Notes:
      - No averaging inside the layer; scaling by batch size happens in the trainer/Model.
      - With softmax, we assume cross-entropy simplification is applied in Model.backward,
        so deltas = grads (i.e., no extra multiply by activation').
    """
    def __init__(self, output_s=None, input_s=None, init="he", activation="relu"):
        super().__init__()
        if input_s is None or output_s is None:
            raise ValueError("Dense: input_s and output_s must be provided.")

        self.input_s = int(input_s)
        self.output_s = int(output_s)
        self.init = init

        # activation function + derivative.
        self.activation, self.activation_prime = get_activation(activation)

        # weights (out, in), biases (out,)
        self.w = init_weights(self.output_s, self.input_s, method=self.init)
        self.b = init_bias(self.output_s, "zero")

        # caches
        self.z_values = None     # (B, out)
        self.a_values = None     # (B, out)
        self.prev_inputs = None    # (B, in)

    def forward(self, inputs):
        x = np.asarray(inputs)
        if x.ndim == 1:           # promote to batch
            x = x[None, :]
        elif x.ndim != 2:
            raise ValueError(f"Dense.forward: expected 1D or 2D input, got {x.ndim}D")

        if x.shape[1] != self.input_s:
            raise ValueError(f"Dense.forward: \
                            input size mismatch: got {x.shape[1]}, expected {self.input_s}")

        self.prev_inputs = x                          # (B, in)
        self.z_values    = x @ self.w.T + self.b[None, :]   # (B, out)
        self.a_values    = self.activation(self.z_values)   # (B, out)
        return self.a_values

    def backward(self, grads):
        g = np.asarray(grads)
        if g.ndim == 1:
            g = g[None, :]
        elif g.ndim != 2:
            raise ValueError(f"Dense.backward: expected 1D or 2D grads, got {g.ndim}D")

        if g.shape[1] != self.output_s:
            raise ValueError(f"Dense.backward: grad size mismatch: got {g.shape[1]}, expected {self.output_s}")

        # Softmax + CE simplification handled in Model.backward -> deltas = grads
        if getattr(self.activation, "__name__", "") == "softmax":
            deltas = g
        else:
            deltas = g * self.activation_prime(self.z_values)  # (B, out)

        # Param grads
        w_grads = deltas.T @ self.prev_inputs   # (out, in)
        b_grads = deltas.sum(axis=0)            # (out,)

        # Input grads
        input_grads = deltas @ self.w           # (B, in)
        return {"w": w_grads, "b": b_grads}, input_grads

    def params(self):
        return {"w": self.w, "b": self.b}

    def apply_grads(self, grads, lr, momentum, clip, weight_decay, state, layer_key):
        key_w = (layer_key, "w")
        key_b = (layer_key, "b")

        if key_w not in state:
            state[key_w] = np.zeros_like(self.w)
        if key_b not in state:
            state[key_b] = np.zeros_like(self.b)

        # weights
        g_w = grads["w"]
        if clip is not None:
            g_w = np.clip(g_w, -clip, +clip)
        v_w = state[key_w]
        v_w = momentum * v_w - lr * g_w
        self.w = self.w * (1 - lr * weight_decay) + v_w
        state[key_w] = v_w

        # biases,
        g_b = grads["b"]
        if clip is not None:
            g_b = np.clip(g_b, -clip, +clip)
        v_b = state[key_b]
        v_b = momentum * v_b - lr * g_b
        self.b += v_b
        state[key_b] = v_b

    def describe(self):
        return f"Dense: {self.output_s} \
            outputs, activation={self.activation.__name__}, input_size={self.input_s}"

    def has_params(self):
        return True
