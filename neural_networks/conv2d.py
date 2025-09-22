from layer_base import Layer
from activations import get_activation
from utils import init_weights, init_bias
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Conv2D(Layer):
    def __init__(self, in_channels, n_kernels, height, width,
                 step_size=1, padding=0, init="he", activation="relu"):
        super().__init__()
        self.in_channels = in_channels
        self.n_kernels   = n_kernels
        self.height      = height
        self.width       = width
        self.step_s      = step_size
        self.padding     = padding
        self.init        = init

        # Activation function and derivative.
        self.activation, self.activation_prime = get_activation(activation)

        # Parameters.
        flat = init_weights(n_kernels, in_channels * height * width, method=self.init)
        self.kernels = flat.reshape(n_kernels, in_channels, height, width)  # (F, C, kh, kw).
        self.biases  = init_bias(n_kernels, "zero")                         # (F,).

        # Caches.
        self.z_values = None        # (B, F, H_out, W_out).
        self.a_values = None        # (B, F, H_out, W_out).
        self.inputs   = None        # Padded input, (B, C, H_pad, W_pad).
        self._windows_flat = None   # (B, n_win, C*kh*kw).
        self._H_out = None
        self._W_out = None

    def forward(self, inputs):
        x = np.asarray(inputs)

        # Normalize to (B, C, H, W).
        if x.ndim == 2:        # (H, W).
            x = x[None, None, :, :]
        elif x.ndim == 3:      # (C, H, W).
            x = x[None, :, :, :]
        elif x.ndim != 4:
            raise ValueError(f"Conv2D.forward: expected 2D/3D/4D, got {x.ndim}D.")

        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Conv2D.forward: in_channels mismatch: got {C}, expected {self.in_channels}.")

        # Optional zero padding.
        if self.padding > 0:
            x = np.pad(
                x,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode="constant"
            )

        self.inputs = x
        _, _, H_pad, W_pad = x.shape

        # Output spatial size.
        H_out = (H_pad - self.height) // self.step_s + 1
        W_out = (W_pad - self.width)  // self.step_s + 1
        if H_out <= 0 or W_out <= 0:
            raise ValueError("Conv2D.forward: output size <= 0; check input, kernel, stride, and padding.")
        self._H_out, self._W_out = H_out, W_out

        # Vectorized window extraction as a view, stride, reorder, then cache flattened windows.
        w = sliding_window_view(x, (self.height, self.width), axis=(2, 3))             # (B, C, H_out, W_out, kh, kw).
        w = w[:, :, ::self.step_s, ::self.step_s, :, :]                                # (B, C, H_out, W_out, kh, kw).
        w = np.transpose(w, (0, 2, 3, 1, 4, 5))                                        # (B, H_out, W_out, C, kh, kw).
        n_win = H_out * W_out
        self._windows_flat = w.reshape(B, n_win, C * self.height * self.width)         # (B, n_win, C*kh*kw).

        # Convolution via einsum + bias.
        k_flat = self.kernels.reshape(self.n_kernels, -1)                               # (F, C*kh*kw).
        z = np.einsum("bnd,kd->bkn", self._windows_flat, k_flat) + self.biases[None, :, None]
        self.z_values = z.reshape(B, self.n_kernels, H_out, W_out)
        self.a_values = self.activation(self.z_values)
        return self.a_values

    def backward(self, grads):
        g = np.asarray(grads)

        # Normalize grads to (B, F, H_out, W_out).
        if g.ndim == 3:  # (F, H_out, W_out).
            g = g[None, ...]
        elif g.ndim != 4:
            raise ValueError(f"Conv2D.backward: expected 3D or 4D grads, got {g.ndim}D.")

        # Local deltas.
        deltas = g * self.activation_prime(self.z_values)  # (B, F, H_out, W_out).
        B, F, H_out, W_out = deltas.shape
        n_win = H_out * W_out
        dflat = deltas.reshape(B, F, n_win)                # (B, F, n_win).

        # Bias grads.
        b_grads = dflat.sum(axis=(0, 2))                   # (F,).

        # Weight grads via cached windows.
        w_grads_flat = np.einsum("bfn,bnd->fd", dflat, self._windows_flat)   # (F, C*kh*kw).
        w_grads = w_grads_flat.reshape(self.kernels.shape)                   # (F, C, kh, kw).

        # Input grads via fast col2im slice-add.
        dx_patches = np.einsum("bfij,fckl->bcijkl", deltas, self.kernels)    # (B, C, H_out, W_out, kh, kw).
        input_grads = np.zeros_like(self.inputs)

        s  = self.step_s
        kh = self.height
        kw = self.width
        for ki in range(kh):
            i_start = ki
            i_end   = i_start + H_out * s
            for kj in range(kw):
                j_start = kj
                j_end   = j_start + W_out * s
                input_grads[:, :, i_start:i_end:s, j_start:j_end:s] += dx_patches[:, :, :, :, ki, kj]

        # Trim padding.
        if self.padding > 0:
            input_grads = input_grads[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return {"w": w_grads, "b": b_grads}, input_grads

    def params(self):
        return {"w": self.kernels, "b": self.biases}

    def has_params(self):
        return True

    def describe(self):
        return (f"Conv2D: {self.n_kernels} kernels, "
                f"in_channels={self.in_channels}, "
                f"size={self.height}x{self.width}, "
                f"stride={self.step_s}, "
                f"padding={self.padding}, "
                f"activation={self.activation.__name__}.")

    def apply_grads(self, grads, lr, momentum, clip, weight_decay, state, layer_key):
        key_w = (layer_key, "w")
        key_b = (layer_key, "b")

        # Initialize momentum slots.
        if key_w not in state:
            state[key_w] = np.zeros_like(self.kernels)
        if key_b not in state:
            state[key_b] = np.zeros_like(self.biases)

        g_w = grads["w"]  # (F, C, kh, kw).
        g_b = grads["b"]  # (F,).

        # Optional grad clipping.
        if clip is not None:
            g_w = np.clip(g_w, -clip, +clip)
            g_b = np.clip(g_b, -clip, +clip)

        # Momentum updates.
        v_w = state[key_w]
        v_b = state[key_b]
        v_w = momentum * v_w - lr * g_w
        v_b = momentum * v_b - lr * g_b

        # L2 weight decay on weights only.
        self.kernels = self.kernels * (1 - lr * weight_decay) + v_w
        self.biases  = self.biases + v_b

        # Save velocity.
        state[key_w] = v_w
        state[key_b] = v_b
