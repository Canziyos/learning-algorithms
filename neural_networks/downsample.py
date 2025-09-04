from layer_base import Layer
import numpy as np
from utils import win_slide_1d, win_slide_2d

class Downsample(Layer):
    """
    downsampling.

    mode: {"max","avg","global_max","global_avg"}
    dim : {1, 2}

    Shapes:
      dim=1
        in : (B, C, L) [also accepts (C, L) or (L)].
        out: local -> (B, C, T_out), global -> (B, C, 1).
      dim=2
        in : (B, C, H, W) [also accepts (C, H, W) or (H, W)]
        out: local -> (B, C, H_out, W_out), global -> (B, C, 1, 1)
    """
    def __init__(self, pool_s, step_s, mode="max", dim=1):
        super().__init__()
        self.pool_s = int(pool_s)
        self.step_s = int(step_s)
        self.mode   = mode
        self.dim    = int(dim)

        self.inputs = None
        self.argmax_indices = None

    def _normalize_shape(self, x):
        x = np.asarray(x)
        if self.dim == 1:
            if x.ndim == 1:      # (L)
                x = x[None, None, :]
            elif x.ndim == 2:    # (C, L)
                x = x[None, :, :]
            elif x.ndim != 3:    # (B, C, L)
                raise ValueError(f"Downsample(dim=1): expected 1D/2D/3D, got {x.ndim}D")
        else:
            if x.ndim == 2:      # (H, W)
                x = x[None, None, :, :]
            elif x.ndim == 3:    # (C, H, W)
                x = x[None, :, :, :]
            elif x.ndim != 4:    # (B, C, H, W)
                raise ValueError(f"Downsample(dim=2): expected 2D/3D/4D, got {x.ndim}D")
        return x

    def forward(self, inputs):
        x = self._normalize_shape(inputs)
        self.inputs = x
        B, C = x.shape[:2]

        if self.dim == 1:
            L = x.shape[2]
            if self.mode == "global_max":
                vals = np.max(x, axis=2); idxs = np.argmax(x, axis=2)
                self.argmax_indices = idxs
                return vals[:, :, None]
            if self.mode == "global_avg":
                self.argmax_indices = None
                return np.mean(x, axis=2)[:, :, None]

            # local
            T_out = (L - self.pool_s) // self.step_s + 1
            out = np.empty((B, C, T_out), dtype=x.dtype)
            arg = np.empty((B, C, T_out), dtype=np.int32) if self.mode == "max" else None
            for b in range(B):
                for c in range(C):
                    w = win_slide_1d(x[b, c], self.pool_s, self.step_s)  # (T_out, pool)
                    if self.mode == "max":
                        out[b, c] = np.max(w, axis=1)
                        arg[b, c] = np.argmax(w, axis=1)
                    else:  # avg
                        out[b, c] = np.mean(w, axis=1)
            self.argmax_indices = arg
            return out

        # dim == 2
        H, W = x.shape[2], x.shape[3]
        if self.mode == "global_max":
            flat = x.reshape(B, C, -1)
            vals = np.max(flat, axis=2); idxs = np.argmax(flat, axis=2)
            self.argmax_indices = idxs
            return vals[:, :, None, None]
        if self.mode == "global_avg":
            self.argmax_indices = None
            return np.mean(x.reshape(B, C, -1), axis=2)[:, :, None, None]

        # local
        H_out = (H - self.pool_s) // self.step_s + 1
        W_out = (W - self.pool_s) // self.step_s + 1
        out = np.empty((B, C, H_out, W_out), dtype=x.dtype)
        arg = np.empty((B, C, H_out, W_out), dtype=np.int32) if self.mode == "max" else None
        for b in range(B):
            for c in range(C):
                w = win_slide_2d(x[b, c], self.pool_s, self.step_s)   # (H_out*W_out, p, p)
                w = w.reshape(H_out, W_out, self.pool_s, self.pool_s)
                if self.mode == "max":
                    out[b, c] = np.max(w, axis=(2, 3))
                    arg[b, c] = np.argmax(w.reshape(H_out, W_out, -1), axis=2)
                else:  # avg
                    out[b, c] = np.mean(w, axis=(2, 3))
        self.argmax_indices = arg
        return out

    def backward(self, grads):
        g = np.asarray(grads)
        x = self.inputs
        B, C = x.shape[:2]

        # ---------- 1D --------- #
        if self.dim == 1:
            if g.ndim == 2: g = g[None, :, :]  # (B,C,T_out) or (B,C,1)
            if g.ndim != 3:
                raise ValueError(f"Downsample.backward(dim=1): expected 2D/3D grads, got {g.ndim}D")

            L = x.shape[2]
            dx = np.zeros_like(x, dtype=float)

            if self.mode == "global_max":
                idxs = self.argmax_indices  # (B,C)
                for b in range(B):
                    for c in range(C):
                        dx[b, c, int(idxs[b, c])] += g[b, c, 0]
                return dx

            if self.mode == "global_avg":
                dx += g / L  # broadcasts (B,C,1) over (B,C,L)
                return dx

            T_out = g.shape[2]
            if self.mode == "max":
                arg = self.argmax_indices  # (B,C,T_out)
                for b in range(B):
                    for c in range(C):
                        starts = np.arange(T_out) * self.step_s
                        pos = starts + arg[b, c]  # (T_out,)
                        np.add.at(dx[b, c], pos, g[b, c])
                return dx

            # ---- avg (vectorized add) ---- #
            p = self.pool_s
            starts = (np.arange(T_out) * self.step_s)[:, None]  # (T_out,1)
            idx = starts + np.arange(p)[None, :]     # (T_out,p)
            for b in range(B):
                for c in range(C):
                    vals = (g[b, c, :, None] / p)      # (T_out,1) -> broadcast to (T_out,p)
                    np.add.at(dx[b, c], idx, vals)
            return dx

        # ---------- 2D ---------- #
        if g.ndim == 3: g = g[None, :, :, :]  # (B,C,H_out,W_out) or (B,C,1,1)
        if g.ndim != 4: raise ValueError(f"Downsample.backward(dim=2): expected 3D/4D grads, got {g.ndim}D")

        H, W = x.shape[2], x.shape[3]
        dx = np.zeros_like(x, dtype=float)

        if self.mode == "global_max":
            idxs = self.argmax_indices  # (B,C) flat in H*W
            for b in range(B):
                for c in range(C):
                    flat = int(idxs[b, c])
                    r, col = flat // W, flat % W
                    dx[b, c, r, col] += g[b, c, 0, 0]
            return dx

        if self.mode == "global_avg":
            area = H * W
            dx += g / area  # (B,C,1,1) broadcast over (B,C,H,W)
            return dx

        H_out, W_out = g.shape[2], g.shape[3]
        p, S = self.pool_s, self.step_s

        if self.mode == "max":
            arg = self.argmax_indices  # (B,C,H_out,W_out)
            for b in range(B):
                for c in range(C):
                    # vectorize inner add-at per output location.
                    r_off = (arg[b, c] // p)
                    c_off = (arg[b, c] % p)
                    rows = (np.arange(H_out) * S)[:, None] + r_off
                    cols = (np.arange(W_out) * S)[None, :] + c_off
                    np.add.at(dx[b, c], (rows, cols), g[b, c])
            return dx

        # ---- avg (vectorized add) ----
        # Build index grids for all patches once.
        r0 = (np.arange(H_out) * S)[:, None] + np.arange(p)[None, :]     # (H_out, p)
        c0 = (np.arange(W_out) * S)[:, None] + np.arange(p)[None, :]     # (W_out, p)
        row_idx = r0[:, None, :, None]        # (H_out, 1, p, 1)
        col_idx = c0[None, :, None, :]        # (1, W_out, 1, p)

        for b in range(B):
            for c in range(C):
                vals = (g[b, c] / (p * p))[:, :, None, None]      # (H_out,W_out,1,1)
                # Broadcast to (H_out, W_out, p, p) and scatter-add.
                np.add.at(dx[b, c], (row_idx, col_idx), vals)
        return dx

    def has_params(self): 
        return False

    def describe(self):
        return f"Downsample(mode={self.mode}, \
            size={self.pool_s}, step={self.step_s}, dim={self.dim})"
