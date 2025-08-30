from layer_base import Layer
import numpy as np
from utils import init_weights, init_bias
from activations import get_activation

class Dense(Layer):
    def __init__(self, output_s=None, input_s=None, init="xavier", activation="relu"):
        self.input_s = input_s
        self.output_s = output_s
        self.init = init

        # activation function + its derivative
        self.activation, self.activation_prime = get_activation(activation)

        # initialize weights and biases
        self.w = init_weights(self.output_s, self.input_s, self.init)
        self.b = init_bias(self.output_s, "zero")

        # storage for forward pass
        self.z_values = None
        self.a_values = None
        self.prev_inputs = None

    def forward(self, inputs):
        if len(inputs) != self.input_s:
            raise ValueError("Input length does not match expected size.")

        self.prev_inputs = np.array(inputs)

        # linear transform
        self.z_values = self.w @ self.prev_inputs + self.b

        # apply activation
        if self.activation and self.activation.__name__ == "softmax":
            self.a_values = self.activation(self.z_values)
        else:
            self.a_values = self.activation(self.z_values)

        return self.a_values

    def backward(self, grads):
        """
        grads: gradient of loss wrt outputs (same shape as a_values).
        Returns: (param_grads, input_grads).
        """
        # 1. deltas
        deltas = grads * self.activation_prime(self.z_values)

        # 2. gradients for weights and biases
        w_grads = np.outer(deltas, self.prev_inputs)
        b_grads = deltas

        # 3. gradients to propagate backwards
        input_grads = self.w.T @ deltas

        return {"w": w_grads, "b": b_grads}, input_grads

    def params(self):
        return {"w": self.w, "b": self.b}

    def apply_grads(self, grads, lr, momentum, clip, weight_decay, state, layer_key):
        key_w = (layer_key, "w")
        key_b = (layer_key, "b")

        # initialize momentum slots if needed
        if key_w not in state:
            state[key_w] = np.zeros_like(self.w)
        if key_b not in state:
            state[key_b] = np.zeros_like(self.b)

        # update weights
        g = grads["w"]
        if clip is not None:
            g = np.clip(g, -clip, +clip)
        v = state[key_w]
        v = momentum * v - lr * g
        self.w = self.w * (1 - lr * weight_decay) + v
        state[key_w] = v

        # update biases
        gb = grads["b"]
        if clip is not None:
            gb = np.clip(gb, -clip, +clip)
        v_b = state[key_b]
        v_b = momentum * v_b - lr * gb
        self.b += v_b
        state[key_b] = v_b

    def describe(self):
        return f"Dense: {self.output_s} outputs, activation={self.activation.__name__}, input_size={self.input_s}"

    def has_params(self):
        return True
