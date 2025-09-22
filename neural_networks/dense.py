from layer_base import Layer
import numpy as np
from utils import init_weights, init_bias
from activations import get_activation

class Dense(Layer):
    """
    Fully connected layer.

    Shapes.
      Forward: inputs (B, in) -> outputs (B, out).
      Backward: grads (B, out) -> input_grads (B, in).

    Notes.
      - No averaging inside the layer; scaling by batch size happens in the trainer/Model.
      - Softmax handling:
          softmax_mode = "ce"      -> use cross-entropy fast-path (deltas = grads), with a runtime check.
          softmax_mode = "general" -> use full softmax Jacobian matvec.
    """

    def __init__(
        self,
        output_s=None,
        input_s=None,
        init="he",
        activation="relu",
        softmax_mode="ce",
        dtype=np.float32,
    ):
        super().__init__()
        if input_s is None or output_s is None:
            raise ValueError("Dense: input_s and output_s must be provided.")

        self.input_s = int(input_s)
        self.output_s = int(output_s)
        self.init = init
        self.dtype = dtype

        # Activation.
        self.activation, self.activation_prime = get_activation(activation)
        self.is_softmax = getattr(self.activation, "__name__", "") == "softmax"
        if self.is_softmax and softmax_mode not in ("ce", "general"):
            raise ValueError("Dense: softmax_mode must be 'ce' or 'general' when activation is softmax.")
        self.softmax_mode = softmax_mode

        # Parameters.
        self.w = init_weights(self.output_s, self.input_s, method=self.init).astype(self.dtype, copy=False)
        self.b = init_bias(self.output_s, "zero").astype(self.dtype, copy=False)

        # Caches.
        self.z_values = None     # (B, out)
        self.a_values = None     # (B, out)
        self.prev_inputs = None  # (B, in)

    def forward(self, inputs):
        x = np.asarray(inputs, dtype=self.dtype)
        if x.ndim == 1:
            x = x[None, :]
        elif x.ndim != 2:
            raise ValueError(f"Dense.forward: expected 1D or 2D input, got {x.ndim}D.")

        if x.shape[1] != self.input_s:
            raise ValueError(f"Dense.forward: input size mismatch: got {x.shape[1]}, expected {self.input_s}.")

        self.prev_inputs = x
        self.z_values    = x @ self.w.T + self.b[None, :]
        self.a_values    = self.activation(self.z_values)
        return self.a_values

    def backward(self, grads):
        if self.prev_inputs is None or self.z_values is None or self.a_values is None:
            raise RuntimeError("Dense.backward called before forward. Call forward() first.")

        g = np.asarray(grads, dtype=self.dtype)
        if g.ndim == 1:
            g = g[None, :]
        elif g.ndim != 2:
            raise ValueError(f"Dense.backward: expected 1D or 2D grads, got {g.ndim}D.")

        if g.shape[1] != self.output_s:
            raise ValueError(f"Dense.backward: grad size mismatch: got {g.shape[1]}, expected {self.output_s}.")

        if self.is_softmax:
            if self.softmax_mode == "ce":
                # CE fast-path expects per-sample grads summing to zero.
                row_sums = g.sum(axis=1)
                if not np.allclose(row_sums, 0.0, atol=1e-6):
                    raise ValueError(
                        "Dense.backward: softmax CE fast-path expects grads with zero row-sum. "
                        "Provide CE logits-grad (y_pred - y_true) or use softmax_mode='general'."
                    )
                deltas = g
            else:
                # Full softmax Jacobian-vector product: J @ g = y * (g - (y*g).sum(axis=1)).
                y = self.a_values
                gy = (g * y).sum(axis=1, keepdims=True)
                deltas = y * (g - gy)
        else:
            deltas = g * self.activation_prime(self.z_values)

        # Parameter grads.
        w_grads = deltas.T @ self.prev_inputs
        b_grads = deltas.sum(axis=0)

        # Input grads.
        input_grads = deltas @ self.w
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

        # Weights.
        g_w = grads["w"]
        if clip is not None:
            g_w = np.clip(g_w, -clip, +clip)
        v_w = state[key_w]
        v_w = momentum * v_w - lr * g_w
        self.w = self.w * (1 - lr * weight_decay) + v_w
        state[key_w] = v_w

        # Biases.
        g_b = grads["b"]
        if clip is not None:
            g_b = np.clip(g_b, -clip, +clip)
        v_b = state[key_b]
        v_b = momentum * v_b - lr * g_b
        self.b += v_b
        state[key_b] = v_b

    def describe(self):
        act_name = getattr(self.activation, "__name__", str(self.activation))
        extra = f", softmax_mode={self.softmax_mode}" if self.is_softmax else ""
        return f"Dense: {self.output_s} outputs, activation={act_name}, input_size={self.input_s}, dtype={self.dtype}{extra}."

    def has_params(self):
        return True
