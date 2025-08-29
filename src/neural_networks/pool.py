from layer_base import Layer
import numpy as np
from utils import win_slide_1d, win_slide_2d

class Pool(Layer):
    def __init__(self, pool_s, step_s, mode="max", dim=1):
        super().__init__()
        self.pool_s = pool_s
        self.step_s = step_s
        self.mode = mode  # "max" or "avg"
        self.dim = dim    # 1 for 1D, 2 for 2D

        # Saved for backward
        self.inputs = None
        self.argmax_indices = None

    def forward(self, inputs):
        inputs = np.array(inputs)
        self.inputs = inputs
        pooled_outputs = []

        if self.dim == 1:
            n_filters, length = inputs.shape
            self.argmax_indices = []
            for f in range(n_filters):
                seq = inputs[f]
                windows = win_slide_1d(seq, self.pool_s, self.step_s)

                if self.mode == "max":
                    pooled = np.max(windows, axis=1)
                    argmax_idx = np.argmax(windows, axis=1)
                    pooled_outputs.append(pooled)
                    self.argmax_indices.append(argmax_idx)
                elif self.mode == "avg":
                    pooled = np.mean(windows, axis=1)
                    pooled_outputs.append(pooled)

        elif self.dim == 2:
            n_filters, H, W = inputs.shape
            self.argmax_indices = []
            for f in range(n_filters):
                seq = inputs[f]
                windows = win_slide_2d(seq, self.pool_s, self.step_s)

                if self.mode == "max":
                    pooled = np.max(windows, axis=(1,2))
                    argmax_idx = np.argmax(
                        windows.reshape(windows.shape[0], -1), axis=1
                    )
                    pooled_outputs.append(pooled)
                    self.argmax_indices.append(argmax_idx)
                elif self.mode == "avg":
                    pooled = np.mean(windows, axis=(1,2))
                    pooled_outputs.append(pooled)

        return np.array(pooled_outputs)

    def backward(self, grads):
        grads = np.asarray(grads)
        input_grads = np.zeros_like(self.inputs, dtype=float)

        if self.dim == 1:
            n_filters, length = self.inputs.shape
            for f in range(n_filters):
                for win_idx, g in enumerate(grads[f]):
                    start = win_idx * self.step_s
                    end = start + self.pool_s

                    if self.mode == "max":
                        max_index = self.argmax_indices[f][win_idx]
                        input_grads[f, start + max_index] += g
                    elif self.mode == "avg":
                        input_grads[f, start:end] += g / self.pool_s

        elif self.dim == 2:
            n_filters, H, W = self.inputs.shape
            win_size = self.pool_s
            for f in range(n_filters):
                out_idx = 0
                for i in range(0, H - win_size + 1, self.step_s):
                    for j in range(0, W - win_size + 1, self.step_s):
                        g = grads[f, out_idx]
                        if self.mode == "max":
                            flat_index = self.argmax_indices[f][out_idx]
                            # convert back to (row, col)
                            row_offset = flat_index // win_size
                            col_offset = flat_index % win_size
                            input_grads[f, i + row_offset, j + col_offset] += g
                        elif self.mode == "avg":
                            input_grads[f, i:i+win_size, j:j+win_size] += g / (win_size * win_size)
                        out_idx += 1

        return input_grads

    def has_params(self):
        return False

    def describe(self):
        return f"Pool: size={self.pool_s}, step={self.step_s}, mode={self.mode}, dim={self.dim}"
