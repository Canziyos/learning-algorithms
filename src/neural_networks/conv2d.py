from layer_base import Layer
from activations import get_activation
from utils import init_weights, init_bias, win_num_2d, win_slide_2d


class Conv2D(Layer):
    def __init__(self, n_kernels, height, width, step_size=1, padding=0, init="xavier", activation="relu"):
        super().__init__()
        self.n_kernels = n_kernels          # number of kernels
        self.height = height                # kernel height
        self.width = width                  # kernel width
        self.step_s = step_size             # stride
        self.padding = padding              # int or (pad_h, pad_w)
        self.init = init

        # Activation
        self.activation, self.activation_prime = get_activation(activation)

        # Create n_filters kernels, each of shape (height × width)
        self.kernels = [
            [init_weights(self.width, 1, method=self.init) for _ in range(self.height)]
            for _ in range(self.n_kernels)
        ]

        # One bias per filter
        self.biases = [init_bias("zero") for _ in range(self.n_kernels)]

        self.z_values = []
        self.a_values = []

    def forward(self, inputs):
        if self.padding > 0:
            row_len = self.width + 2*self.padding

            # Build top padding rows
            top_rows = [[0]*row_len for _ in range(self.padding)]

            # Pad each input row horizontally
            padded_rows = []
            for row in inputs:
                new_row = [0]*self.padding + row + [0]*self.padding
                padded_rows.append(new_row)

            # Build bottom padding rows
            bottom_rows = [[0]*row_len for _ in range(self.padding)]

            # Concatenate
            self.inputs = top_rows + padded_rows + bottom_rows
        else:
            self.inputs = inputs

        out_h, out_w = win_num_2d(
            len(inputs),        # unpadded input height
            len(inputs[0]),     # unpadded input width
            self.padding,       # padding
            self.height,        # kernel height
            self.width,         # kernel width
            self.step_s
        )

        self.z_values = [
            [[0.0 for _ in range(out_w)] for _ in range(out_h)]
            for _ in range(self.n_kernels)
        ]
        self.a_values = [
            [[0.0 for _ in range(out_w)] for _ in range(out_h)]
            for _ in range(self.n_kernels)
        ]

        for f in range(self.n_kernels):
            for row in range(out_h):
                for col in range(out_w):
                    tl_row_idx = row * self.step_s
                    tl_col_idx = col * self.step_s
                    br_row_idx = tl_row_idx + self.height
                    br_col_idx = tl_col_idx + self.width
                    window = [
                        self.inputs[r][tl_col_idx:br_col_idx]
                        for r in range(tl_row_idx, br_row_idx)
                    ]

                    w = self.kernels[f]
                    b = self.biases[f]

                    z = 0.0
                    for i in range(self.height):
                        for j in range(self.width):
                            z += window[i][j] * w[i][j]

                    z += b
                    self.z_values[f][row][col] = z

                    self.a_values[f][row][col] = self.activation(z)
        return self.a_values

    def backward(self, grads):
        for f in range(self.n_kernels):
            if any(len(grads[f][r]) != len(self.a_values[f][r]) for r in range(len(self.a_values[f]))):
                raise ValueError("Shape mismatch inside feature maps.")
        deltas = [[[0.0 for _ in range(len(self.a_values[f][r]))]
                for r in range(len(self.a_values[f]))]
                for f in range(self.n_kernels)
                ]
        w_grads = [[[0.0 for _ in range(self.width)]
                for r in range(self.height)]
                for f in range(self.n_kernels)
                ]
        b_grads = [0.0 for _ in range(self.n_kernels)]
        input_grads = [[0.0 for _ in range(len(self.inputs[0]))] for _ in range(len(self.inputs))]

        for f in range(self.n_kernels):
            for row in range(len(self.a_values[f])):
                for col in range(len(self.a_values[f][row])):
                    z = self.z_values[f][row][col]
                    g = grads[f][row][col]
                    delta = g * self.activation_prime(z)
                    deltas[f][row][col] = delta
                    b_grads[f] += delta

                    tl_row = row*self.step_s
                    tl_col = col*self.step_s
                    br_row = tl_row+self.height
                    br_col = tl_col + self.width
                    for i in range(tl_row, br_row):
                        for j in range(tl_col, br_col):
                            w_grads[f][i - tl_row][j - tl_col] += delta * self.inputs[i][j]
                            input_grads[i][j] += delta * self.kernels[f][i - tl_row][j - tl_col]
        if self.padding > 0:
            input_grads = [row[self.padding: -self.padding] for row in input_grads[self.padding:-self.padding]]

        return {"w": w_grads, "b": b_grads}, input_grads

    def has_params(self):
        return True
    




c = Conv2D(2, 2, 3, 1, 2)
inputs = [[1, 3], [2, 5]]
outs = c.forward(inputs)
grads = [[[1.0 for _ in row] for row in fmap] for fmap in outs]
c.backward(grads)
 
