import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return max(0.0, z)

def relu_prime(z):
    return 1.0 if z > 0 else 0.0

def identity(z):
    return z

def identity_prime(z):
    return 1.0

def tanh(z):
    return math.tanh(z)

def tanh_prime(z):
    t = math.tanh(z)
    return 1 - t**2

def softmax(z_values):
    # Subtract max (numerical stability).
    max_z = max(z_values)
    exp_vals = [math.exp(z - max_z) for z in z_values]
    sum_exp = sum(exp_vals)
    return [val / sum_exp for val in exp_vals]

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
