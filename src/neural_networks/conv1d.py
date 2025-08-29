from layer_base import Layer
from utils import init_weights, init_bias, win_num, dprint
from activations import get_activation

class Conv1D(Layer):
    def __init__(self, n_filters=None, filter_s=None, step_s=1, padding=0,
                 init="xavier", activation="relu"):
        super().__init__()
        self.n_filters = n_filters          # number of kernels.
        self.filter_s = filter_s            # length of each kernel.
        self.step_s = step_s
        self.padding = padding
        self.init = init                    # weight init method (string).
        

        # activation function + its derivative.
        self.activation, self.activation_prime = get_activation(activation)

        # initialize filters and biases.
        self.filters = [
            init_weights(self.filter_s, 1, method=self.init)  # fan_out=1 (textbook style).
            for _ in range(self.n_filters)
        ]
        self.biases = [init_bias("zero") for _ in range(self.n_filters)]

        # checked the effect of identical inits.
        #self.filters = [[0.5, -0.5] for _ in range(self.n_filters)]
        #self.biases = [0.0 for _ in range(self.n_filters)]
        
        # storage for forward pass.
        self.z_values = []
        self.a_values = []

    def forward(self, inputs):
        # 1. Pad input if needed.
        if self.padding > 0:
            zeros_left = [0] * self.padding
            zeros_right = [0] * self.padding
            padded_inputs = zeros_left + inputs + zeros_right
        else:
            padded_inputs = inputs

        # store padded input for backward.
        self.inputs = padded_inputs

        # 2. Compute number of windows (output length).
        out_len = win_num(len(inputs), self.padding, self.filter_s, self.step_s)

        # reset storages.
        self.z_values = [[] for _ in range(self.n_filters)]
        self.a_values = [[] for _ in range(self.n_filters)]

        dprint(2, f"\n[Conv1D Forward] inputs={inputs}")
        dprint(2, f"Padded={padded_inputs}, out_len={out_len}")

        # 3. Loop over windows.
        for pos in range(out_len):
            start = pos * self.step_s
            end = start + self.filter_s
            window = padded_inputs[start:end]

            for f in range(self.n_filters):
                w = self.filters[f]
                b = self.biases[f]
                z = sum(window[i] * w[i] for i in range(self.filter_s)) + b
                self.z_values[f].append(z)

        # 4. Apply activation.
        for f in range(self.n_filters):
            self.a_values[f] = [self.activation(z) for z in self.z_values[f]]
            dprint(2, f"Filter {f} weights={self.filters[f]}, bias={self.biases[f]}")
            dprint(2, f"Filter {f} z={self.z_values[f]}")
            dprint(2, f"Filter {f} a={self.a_values[f]}")

        return self.a_values

    def backward(self, grads):
        deltas = [[] for _ in range(self.n_filters)]
        w_grads = [[0.0 for _ in range(self.filter_s)] for _ in range(self.n_filters)]
        b_grads = [0.0 for _ in range(self.n_filters)]
        input_grads = [0.0 for _ in range(len(self.inputs))]
        
        if len(grads) != self.n_filters or any(len(g) != len(self.a_values[f]) for f, g in enumerate(grads)):
            raise ValueError("Shape mismatch: grads must have same shape as a_values.")

        # 1. Compute deltas.
        for f in range(self.n_filters):
            for p in range(len(self.z_values[f])):
                z = self.z_values[f][p]
                g = grads[f][p]
                delta = g * self.activation_prime(z)
                deltas[f].append(delta)
                b_grads[f] += delta

        # 2. Compute weight gradients.
        for f in range(self.n_filters):
            for p in range(len(self.z_values[f])):
                start = p * self.step_s
                end = start + self.filter_s
                window = self.inputs[start:end]
                for i in range(self.filter_s):
                    w_grads[f][i] += deltas[f][p] * window[i]

        # 3. Compute input gradients.
        for f in range(self.n_filters):
            for p in range(len(self.z_values[f])):
                start = p * self.step_s
                end = start + self.filter_s
                for i in range(self.filter_s):
                    input_grads[start+i] += deltas[f][p] * self.filters[f][i]

        # Trim padding.
        if self.padding > 0:
            input_grads = input_grads[self.padding: -self.padding]

        dprint(2, "\n[Conv1D Backward]")
        for f in range(self.n_filters):
            dprint(2, f"Filter {f} deltas={deltas[f]}")
            dprint(2, f"Filter {f} weight grads={w_grads[f]}")
            dprint(2, f"Filter {f} bias grad={b_grads[f]}")
        dprint(2, f"Input grads={input_grads}")

        return {"w": w_grads, "b": b_grads}, input_grads

    def params(self):
        return {"w": self.filters, "b": self.biases}

    def apply_grads(self, grads, lr, momentum, clip, weight_decay, state, layer_key):
        for f in range(self.n_filters):
            key_w = (layer_key, f, "w")
            key_b = (layer_key, f, "b")

            if key_w not in state:
                state[key_w] = [0.0 for _ in self.filters[f]]
            if key_b not in state:
                state[key_b] = 0.0

            # update filter weights.
            for i in range(self.filter_s):
                g = grads["w"][f][i]
                if clip is not None:
                    g = max(min(g, clip), -clip)
                v = state[key_w][i]
                v = momentum * v - lr * g
                self.filters[f][i] = self.filters[f][i] * (1 - lr * weight_decay) + v
                state[key_w][i] = v

            # update bias.
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
        return f"Conv1D: {self.n_filters} filters, size={self.filter_s}, step_size={self.step_s}, padding={self.padding}, activation={self.activation.__name__}"
