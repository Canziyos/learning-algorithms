import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0.0, z)

def relu_prime(z):
    return (z>0).astype(float)

def identity(z):
    return z

def identity_prime(z):
    return np.ones_like(z)

def tanh(z):
    return np.tanh(z)

def tanh_prime(z):
    t = np.tanh(z)
    return 1 - t**2

def softmax(z_values):
    z = np.asarray(z_values)
    # subtract row-wise (last-axis)-
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_vals = np.exp(z)
    sum_exp = np.sum(exp_vals, axis=-1, keepdims=True)
    return exp_vals / sum_exp

def get_activation(name):
    """
    Return (activation, activation_prime) tuple by name.
    Special case: softmax has no derivative here (handled in loss).
    """
    name = name.lower()
    if name == "sigmoid":
        return sigmoid, sigmoid_prime
    elif name == "relu":
        return relu, relu_prime
    elif name == "tanh":
        return tanh, tanh_prime
    elif name == "identity":
        return identity, identity_prime
    elif name == "softmax":
        return softmax, None
    else:
        raise ValueError(f"Unknown activation: {name}")
