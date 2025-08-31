from layer_base import Layer
from activations import get_activation
from utils import init_weights, init_bias, win_slide_2d
import numpy as np

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

        # activation function + derivative.
        self.activation, self.activation_prime = get_activation(activation)

        # Kernels: (n_kernels, in_channels, height, width).
        flat = init_weights(n_kernels, in_channels * height * width, method=self.init)
        self.kernels = flat.reshape(n_kernels, in_channels, height, width)

        # One bias per kernel.
        self.biases = init_bias(n_kernels, "zero")

        # Storage.
        self.z_values = None   # (B, n_k, H_out, W_out)
        self.a_values = None   # (B, n_k, H_out, W_out)
        self.inputs = None   # (B, C, H_pad, W_pad)

    def forward(self, inputs):
        x = np.asarray(inputs)

        # Normalize to (B, C, H, W).
        if x.ndim == 2:        # (H, W)
            x = x[None, None, :, :]
        elif x.ndim == 3:      # (C, H, W)
            x = x[None, :, :, :]
        elif x.ndim != 4:
            raise ValueError(f"Conv2D.forward: expected 2D/3D/4D, got {x.ndim}D")

        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Conv2D.forward: in_channels mismatch: got {C}, expected {self.in_channels}")

        if self.padding > 0:
            x = np.pad(
                x,
                ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)),
                mode="constant"
            )

        self.inputs = x
        _, _, H_pad, W_pad = x.shape

        out_h = (H_pad - self.height) // self.step_s + 1
        out_w = (W_pad - self.width) // self.step_s + 1

        # Extract windows per batch & channel → (B, n_windows, C*h*w).
        windows = []
        for b in range(B):
            channel_windows = []
            for c in range(C):
                w = win_slide_2d(x[b, c], self.height, self.step_s)  # (n_windows, h, w)
                channel_windows.append(w)
            channel_windows = np.stack(channel_windows, axis=1)      # (n_windows, C, h, w)
            windows.append(channel_windows.reshape(channel_windows.shape[0], -1))
        windows = np.stack(windows, axis=0)                           # (B, n_windows, C*h*w)

        # Kernels: (n_k, C*h*w).
        kernels_flat = self.kernels.reshape(self.n_kernels, -1)

       # kernels_flat: (n_k, C*h*w)
        z = np.einsum("bnd,kd->bkn", windows, kernels_flat) + self.biases[None, :, None]  # (B, n_k, n_windows)

        self.z_values = z.reshape(B, self.n_kernels, out_h, out_w)
        self.a_values = self.activation(self.z_values)
        return self.a_values


    def backward(self, grads):
        g = np.asarray(grads)

        # Normalize grads to (B, n_k, H_out, W_out).
        if g.ndim == 3:  # (n_k, H_out, W_out)
            g = g[None, ...]
        elif g.ndim != 4:
            raise ValueError(f"Conv2D.backward: expected 3D or 4D grads, got {g.ndim}D")

        # Deltas with same shape as z_values.
        deltas = g * self.activation_prime(self.z_values)
        B, n_k, H_out, W_out = deltas.shape

        # Bias grads: sum over batch & spatial.
        b_grads = np.sum(deltas, axis=(0, 2, 3))  # (n_k,)

        # Weight grads and input grads.
        w_grads = np.zeros_like(self.kernels)
        input_grads = np.zeros_like(self.inputs)

        for b in range(B):
            for i in range(H_out):
                for j in range(W_out):
                    win = self.inputs[b, :,
                                      i*self.step_s:i*self.step_s+self.height,
                                      j*self.step_s:j*self.step_s+self.width]   # (C,h,w)
                    for f in range(self.n_kernels):
                        d = deltas[b, f, i, j]
                        w_grads[f] += d * win
                        input_grads[b, :,
                                    i*self.step_s:i*self.step_s+self.height,
                                    j*self.step_s:j*self.step_s+self.width] += d * self.kernels[f]

        # Trim padding on input grads.
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
                f"activation={self.activation.__name__}")

    def apply_grads(self, grads, lr, momentum, clip, weight_decay, state, layer_key):

        key_w = (layer_key, "w")
        key_b = (layer_key, "b")

        # init momentum slots
        if key_w not in state:
            state[key_w] = np.zeros_like(self.kernels)
        if key_b not in state:
            
            state[key_b] = np.zeros_like(self.biases)

        g_w = grads["w"]           # shape (n_kernels, in_channels, h, w)
        g_b = grads["b"]           # shape (n_kernels,)

        # optional grad clipping.
        if clip is not None:
            g_w = np.clip(g_w, -clip, +clip)
            g_b = np.clip(g_b, -clip, +clip)

        # momentum updates.
        v_w = state[key_w]
        v_b = state[key_b]
        v_w = momentum * v_w - lr * g_w
        v_b = momentum * v_b - lr * g_b

        # L2 only on weights.
        self.kernels = self.kernels * (1 - lr * weight_decay) + v_w
        self.biases  = self.biases + v_b

        # save velocity.
        state[key_w] = v_w
        state[key_b] = v_b
