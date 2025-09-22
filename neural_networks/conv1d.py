from layer_base import Layer
from utils import init_weights, init_bias, win_slide_1d
from activations import get_activation
import numpy as np

class Conv1D(Layer):
    def __init__(self, n_kernels=None, kernel_size=None, step_s=1, padding=0,
                 init="xavier", activation="relu"):
        super().__init__()
        self.n_k = n_kernels           # number of kernels.
        self.k_s = kernel_size         # kernel length.
        self.step_s = step_s
        self.padding = padding
        self.init = init

        # activation + derivative.
        self.activation, self.activation_prime = get_activation(activation)

        # kernels: (n_k, k_s), biases: (n_k,).
        self.kernels = init_weights(self.n_k, self.k_s, method=self.init)
        self.biases  = init_bias(self.n_k, "zero")

        # caches.
        self.inputs = None     # padded inputs (B, L_pad).
        self.z_values = None     # (B, n_k, T).
        self.a_values = None     # (B, n_k, T).
        self.windows = None     # (B, T, k_s) for weight grads.
        self._idx = None     # (T, k_s) scatter index for backward.

    def forward(self, inputs):
        x = np.asarray(inputs)

        # Normalize to (B, L).
        if x.ndim == 1:
            x = x[None, :]
        elif x.ndim != 2:
            raise ValueError(f"Conv1D.forward: expected 1D or 2D, got {x.ndim}D.")

        # Pad along time axis.
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (self.padding, self.padding)), mode="constant")
        self.inputs = x  # (B, L_pad).

        # Sliding windows per sample -> (B, T, k_s).
        self.windows = np.stack([win_slide_1d(x[b], self.k_s, self.step_s)
                                 for b in range(x.shape[0])], axis=0)  # (B, T, k_s).
        B, T, k_s = self.windows.shape
        assert k_s == self.k_s, f"Window size mismatch: {k_s} vs {self.k_s}."

        # Convolution: z[b, n, t] = sum_k kernels[n,k] * windows[b,t,k].
        z = np.einsum("btk,nk->bnt", self.windows, self.kernels) + self.biases[None, :, None]

        # Cache scatter indices for backward.
        starts = np.arange(T) * self.step_s                      # (T,).
        self._idx = starts[:, None] + np.arange(self.k_s)[None, :]  # (T, k_s).
        assert self._idx.max() < self.inputs.shape[1], \
            f"Scatter index {self._idx.max()} out of bounds for L_pad={self.inputs.shape[1]}."

        self.z_values = z
        self.a_values = self.activation(z)  # (B, n_k, T).
        return self.a_values

    def backward(self, grads):
        g = np.asarray(grads)

        # Normalize grads to (B, n_k, T).
        if g.ndim == 2:
            g = g[None, ...]
        elif g.ndim != 3:
            raise ValueError(f"Conv1D.backward: expected 2D or 3D grads, got {g.ndim}D.")

        B = g.shape[0]

        # Deltas, honoring softmax special case if ever used here.
        if getattr(self.activation, "__name__", "") == "softmax":
            deltas = g
        else:
            deltas = g * self.activation_prime(self.z_values)  # (B, n_k, T).

        # Sanity checks.
        _, n_k, T = deltas.shape
        assert self.windows is not None and self._idx is not None, "Forward caches missing."
        assert self.windows.shape == (B, T, self.k_s), \
            f"windows {self.windows.shape} vs expected {(B, T, self.k_s)}."
        assert self.kernels.shape == (self.n_k, self.k_s), f"kernels {self.kernels.shape}."

        # Weight and bias gradients.
        w_grads = np.einsum("bnt,btk->nk", deltas, self.windows)   # (n_k, k_s).
        b_grads = deltas.sum(axis=(0, 2))                          # (n_k,).

        # Fully vectorized input grads via scatter-add.
        # contribs[b, t, k] = sum_f deltas[b, f, t] * kernels[f, k].
        contribs = np.einsum("bnt,nk->btk", deltas, self.kernels)   # (B, T, k_s).

        input_grads = np.zeros_like(self.inputs, dtype=float)       # (B, L_pad).
        B, L_pad = input_grads.shape
        idx = self._idx                                             # (T, k_s).
        assert idx.max() < L_pad, f"Scatter index {idx.max()} out of bounds for L_pad={L_pad}."

        # Build broadcastable indices for np.add.at and scatter.
        b_idx   = np.arange(B)[:, None, None]                       # (B, 1, 1).
        tk_idx  = idx[None, :, :]                                   # (1, T, k_s).
        np.add.at(input_grads, (b_idx, tk_idx), contribs)           # scatter-add.

        # Trim padding on inputs.
        if self.padding > 0:
            input_grads = input_grads[:, self.padding:-self.padding]

        return {"w": w_grads, "b": b_grads}, input_grads

    def params(self):
        return {"w": self.kernels, "b": self.biases}

    def apply_grads(self, grads, lr, momentum, clip, weight_decay, state, layer_key):
        for f in range(self.n_k):
            key_w = (layer_key, f, "w")
            key_b = (layer_key, f, "b")

            if key_w not in state:
                state[key_w] = np.zeros_like(self.kernels[f])
            if key_b not in state:
                state[key_b] = 0.0

            # weights.
            g_w = grads["w"][f]
            if clip is not None:
                g_w = np.clip(g_w, -clip, +clip)
            v_w = state[key_w]
            v_w = momentum * v_w - lr * g_w
            self.kernels[f] = self.kernels[f] * (1 - lr * weight_decay) + v_w
            state[key_w] = v_w

            # biases.
            g_b = grads["b"][f]
            if clip is not None:
                g_b = np.clip(g_b, -clip, +clip)
            v_b = state[key_b]
            v_b = momentum * v_b - lr * g_b
            self.biases[f] += v_b
            state[key_b] = v_b

    def has_params(self):
        return True

    def describe(self):
        return (f"Conv1D: {self.n_k} filters, size={self.k_s}, "
                f"stride={self.step_s}, padding={self.padding}, "
                f"activation={self.activation.__name__}")
