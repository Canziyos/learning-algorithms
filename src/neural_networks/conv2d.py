from layer_base import Layer
from activations import get_activation
from utils import init_weights, init_bias, win_num_2d, win_slide_2d


class Conv2D(Layer):
    def __init__(self, in_channels, n_kernels, height, width,
                 step_size=1, padding=0, init="xavier", activation="relu"):
        super().__init__()
        self.in_channels = in_channels      # number of input channels
        self.n_kernels = n_kernels          # number of output kernels
        self.height = height                # kernel height
        self.width = width                  # kernel width
        self.step_s = step_size             # stride
        self.padding = padding              # int or (pad_h, pad_w)
        self.init = init

        # Activation
        self.activation, self.activation_prime = get_activation(activation)

        # Kernels: shape [out_channels][in_channels][height][width]
        self.kernels = [
            [
                [init_weights(self.width, 1, method=self.init) for _ in range(self.height)]
                for _ in range(self.in_channels)
            ]
            for _ in range(self.n_kernels)
        ]

        # One bias per output kernel
        self.biases = [init_bias("zero") for _ in range(self.n_kernels)]

        # Storage
        self.z_values = []
        self.a_values = []

    def forward(self, inputs):
        """
        inputs: list of [in_channels][H][W]
        """
        # Pad each channel separately
        if self.padding > 0:
            padded_inputs = []
            for c in range(self.in_channels):
                row_len = len(inputs[c][0]) + 2 * self.padding

                # Top padding rows
                top_rows = [[0] * row_len for _ in range(self.padding)]

                # Pad each row horizontally
                padded_rows = []
                for row in inputs[c]:
                    new_row = [0] * self.padding + row + [0] * self.padding
                    padded_rows.append(new_row)

                # Bottom padding rows
                bottom_rows = [[0] * row_len for _ in range(self.padding)]

                # Full padded channel
                padded_inputs.append(top_rows + padded_rows + bottom_rows)
            self.inputs = padded_inputs
        else:
            self.inputs = inputs

        # Output size
        H, W = len(inputs[0]), len(inputs[0][0])   # original unpadded size
        out_h, out_w = win_num_2d(H, W, self.padding,
                                self.height, self.width, self.step_s)

        # Storage
        self.z_values = [
            [[0.0 for _ in range(out_w)] for _ in range(out_h)]
            for _ in range(self.n_kernels)
        ]
        self.a_values = [
            [[0.0 for _ in range(out_w)] for _ in range(out_h)]
            for _ in range(self.n_kernels)
        ]

        # Convolution per kernel
        for f in range(self.n_kernels):
            for row in range(out_h):
                for col in range(out_w):
                    tl_row = row * self.step_s
                    tl_col = col * self.step_s
                    br_row = tl_row + self.height
                    br_col = tl_col + self.width

                    z = 0.0
                    # Sum over channels
                    for c in range(self.in_channels):
                        window = [
                            self.inputs[c][r][tl_col:br_col]
                            for r in range(tl_row, br_row)
                        ]
                        w = self.kernels[f][c]
                        for i in range(self.height):
                            for j in range(self.width):
                                z += window[i][j] * w[i][j]

                    z += self.biases[f]
                    self.z_values[f][row][col] = z
                    self.a_values[f][row][col] = self.activation(z)

        return self.a_values

    def backward(self, grads):
        # --- check shapes ---
        for f in range(self.n_kernels):
            if any(len(grads[f][r]) != len(self.a_values[f][r]) for r in range(len(self.a_values[f]))):
                raise ValueError("Shape mismatch inside feature maps.")

        # --- Allocate containers ---
        deltas = [
            [[0.0 for _ in range(len(self.a_values[f][r]))] for r in range(len(self.a_values[f]))]
            for f in range(self.n_kernels)
        ]
        # kernels shape: [out][in][h][w]
        w_grads = [
            [
                [[0.0 for _ in range(self.width)] for _ in range(self.height)]
                for _ in range(self.in_channels)
            ]
            for _ in range(self.n_kernels)
        ]
        b_grads = [0.0 for _ in range(self.n_kernels)]

        # input_grads shaped like self.inputs: [in_channels][H_padded][W_padded]
        input_grads = [
            [[0.0 for _ in range(len(self.inputs[c][0]))] for _ in range(len(self.inputs[c]))]
            for c in range(self.in_channels)
        ]

        # --- Backprop main loop ---
        for f in range(self.n_kernels):
            for row in range(len(self.a_values[f])):
                for col in range(len(self.a_values[f][row])):
                    z = self.z_values[f][row][col]
                    g = grads[f][row][col]
                    delta = g * self.activation_prime(z)
                    deltas[f][row][col] = delta
                    b_grads[f] += delta

                    # Receptive field bounds
                    tl_row = row * self.step_s
                    tl_col = col * self.step_s
                    br_row = tl_row + self.height
                    br_col = tl_col + self.width

                    # Loop over input channels
                    for c in range(self.in_channels):
                        for i in range(tl_row, br_row):
                            for j in range(tl_col, br_col):
                                # Update weight grads
                                w_grads[f][c][i - tl_row][j - tl_col] += delta * self.inputs[c][i][j]
                                # Accumulate into input grads
                                input_grads[c][i][j] += delta * self.kernels[f][c][i - tl_row][j - tl_col]

        # --- Trim padding off input_grads ---
        if self.padding > 0:
            trimmed = []
            for c in range(self.in_channels):
                slice_rows = input_grads[c][self.padding:-self.padding]
                trimmed_c = [row[self.padding:-self.padding] for row in slice_rows]
                trimmed.append(trimmed_c)
            input_grads = trimmed

        return {"w": w_grads, "b": b_grads}, input_grads
    
    def params(self):
        return {"w": self.kernels, "b": self.biases}
    
    def apply_grads(self, grads, lr, momentum, clip, weight_decay, state, layer_key):
        for f in range(self.n_kernels):
            for c in range(self.in_channels):
                key_w = (layer_key, f, c, "w")
                if key_w not in state:
                    # Initialize momentum state for each weight in this slice
                    state[key_w] = [
                        [0.0 for _ in range(self.width)]
                        for _ in range(self.height)
                    ]

                for i in range(self.height):
                    for j in range(self.width):
                        g = grads["w"][f][c][i][j]
                        if clip is not None:
                            g = max(min(g, clip), -clip)

                        v = state[key_w][i][j]
                        v = momentum * v - lr * g

                        # Weight decay (L2 regularization)
                        self.kernels[f][c][i][j] = (
                            self.kernels[f][c][i][j] * (1 - lr * weight_decay) + v
                        )
                        state[key_w][i][j] = v

            # Bias update
            key_b = (layer_key, f, "b")
            if key_b not in state:
                state[key_b] = 0.0

            gb = grads["b"][f]
            if clip is not None:
                gb = max(min(gb, clip), -clip)

            v_b = state[key_b]
            v_b = momentum * v_b - lr * gb
            self.biases[f] += v_b
            state[key_b] = v_b

    def has_params(self):
        return True

    def describe(self):
        return (f"Conv2D: {self.n_kernels} kernels, "
                f"in_channels={self.in_channels}, "
                f"size={self.height}x{self.width}, "
                f"stride={self.step_s}, "
                f"padding={self.padding}, "
                f"activation={self.activation.__name__}")
    
