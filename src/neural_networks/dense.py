from layer_base import Layer
from neuron import Neuron
from utils import init_weights, init_bias, dprint
from activations import get_activation

class Dense(Layer):
    def __init__(self, input_s=None, n_neurons=None,
                 init="xavier",
                 activation="relu"):
        self.input_s = input_s
        self.n_neurons = n_neurons
        self.z_values = []
        self.a_values = []

        # get activation and its derivative from utils.
        self.activation, self.activation_prime = get_activation(activation)

        # build neurons.
        self.neurons = []
        for _ in range(self.n_neurons):
            weights = init_weights(input_s, n_neurons, method=init)
            bias = init_bias(method="zero")
            neuron = Neuron(input_s=input_s)
            neuron.weights = weights
            neuron.bias = bias
            self.neurons.append(neuron)

    def forward(self, inputs):
        self.prev_inputs = inputs[:]
        self.z_values = [n.forward(inputs) for n in self.neurons]

        if self.activation and self.activation.__name__ == "softmax":
            # softmax applied over vector.
            self.a_values = self.activation(self.z_values)
        else:
            self.a_values = [self.activation(z) for z in self.z_values]

        dprint(2, "\n[Dense Forward]")
        dprint(2, f"Inputs={inputs}")
        dprint(2, f"z_values={self.z_values}")
        dprint(2, f"a_values={self.a_values}")

        return self.a_values
    
    def backward(self, grads):
        """
        grads: gradient of loss wrt outputs (same length as a_values).
        Returns: (param_grads, input_grads).
        """
        # 1. deltas.
        deltas = [g * self.activation_prime(z) for g, z in zip(grads, self.z_values)]

        # 2. weight & bias grads.
        w_grads = [[delta * a for a in self.prev_inputs] for delta in deltas]
        b_grads = deltas[:]

        # 3. input grads (propagate backwards).
        input_grads = [0.0 for _ in range(self.input_s)]
        for j, delta in enumerate(deltas):
            for i in range(self.input_s):
                input_grads[i] += delta * self.neurons[j].weights[i]

        dprint(2, "\n[Dense Backward]")
        dprint(2, f"Incoming grads={grads}")
        dprint(2, f"Deltas={deltas}")
        dprint(2, f"Weight grads={w_grads}")
        dprint(2, f"Bias grads={b_grads}")
        dprint(2, f"Input grads={input_grads}")

        return {"w": w_grads, "b": b_grads}, input_grads

    def params(self):
        return {"w": [n.weights for n in self.neurons],
                "b": [n.bias for n in self.neurons]}

    def apply_grads(self, grads, lr, momentum, clip, weight_decay, state, layer_key):
        for j, neuron in enumerate(self.neurons):
            key_w = (layer_key, j, "w")
            key_b = (layer_key, j, "b")

            # initialize momentum slots if needed.
            if key_w not in state:
                state[key_w] = [0.0 for _ in neuron.weights]
            if key_b not in state:
                state[key_b] = 0.0

            # update weights.
            for i in range(len(neuron.weights)):
                g = grads["w"][j][i]
                if clip is not None:
                    g = max(min(g, clip), -clip)
                v = state[key_w][i]
                v = momentum * v - lr * g
                neuron.weights[i] = neuron.weights[i] * (1 - lr * weight_decay) + v
                state[key_w][i] = v

            # update bias.
            gb = grads["b"][j]
            if clip is not None:
                gb = max(min(gb, clip), -clip)
            v_b = state[key_b]
            v_b = momentum * v_b - lr * gb
            neuron.bias += v_b
            state[key_b] = v_b

    def describe(self):
        return f"Dense: {self.n_neurons} neurons, activation={self.activation.__name__}, input_size={self.input_s}"

    def has_params(self):
        return True
