import random

class Neuron:
    def __init__(self, input_s=None,):

        self.input_s = input_s

        # The following will be set by DenseLayer
        self.weights = []  
        self.bias = 0.0 


    def forward(self, inputs):
        if len(inputs) != self.input_s:
            raise ValueError("Input length does not match number of weights!")

        # Weighted sum.
        z = sum(inputs[i] * self.weights[i] for i in range(self.input_s)) + self.bias

        return z

