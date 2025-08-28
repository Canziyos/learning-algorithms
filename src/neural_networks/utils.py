import math, random

# Debug levels:
# 0 = silent
# 1 = epoch summaries
# 2 = full step-by-step chaos
DEBUG_LEVEL = 1  

def dprint(level, *args, **kwargs):
    """Debug print controlled by DEBUG_LEVEL."""
    if DEBUG_LEVEL >= level:
        print(*args, **kwargs)


def MSE(y_true, y_pred):
    """
    Mean squared error loss.
    y_true and y_pred must be non-empty and equal length.
    """
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("MSE: y_true and y_pred must be non-empty and equal length.")
    n = len(y_true)
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n

def output_layer_delta(y_true, y_pred, z_values, activation_prime):
    """
    Delta for the output layer (using MSE).
    Scales error by activation derivative.
    """
    n = len(y_true)
    return [(2/n) * (yp - yt) * activation_prime(z)
            for yp, yt, z in zip(y_pred, y_true, z_values)]

def output_gradient(delta, prev_activations):
    """
    Compute weight and bias gradients for the output layer.
    """
    weight_grads = [[delta[j] * a for a in prev_activations]
                    for j in range(len(delta))]
    bias_grads = delta[:]  # Copy.
    return weight_grads, bias_grads

def hidden_layer_delta(z_values, activation_prime, next_layer_weights, next_layer_delta):
    """
    Compute deltas for a hidden layer.
    Each neuron’s delta depends on downstream weights and deltas.
    """
    deltas = []
    for i, z in enumerate(z_values):
        downstream = sum(next_layer_weights[j][i] * next_layer_delta[j] 
                         for j in range(len(next_layer_delta)))
        deltas.append(activation_prime(z) * downstream)
    return deltas

def cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Cross-entropy loss for multi-class classification.
    y_true: one list of true labels (e.g., [0,1,0]).
    y_pred: list of predicted probabilities (softmax output).
    eps: small value to avoid log(0).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("cross_entropy: y_true and y_pred must be the same length.")

    # Clip predictions to avoid log(0).
    y_pred = [min(max(p, eps), 1 - eps) for p in y_pred]

    return -sum(t * math.log(p) for t, p in zip(y_true, y_pred))

def batch_backward(net, batch, loss_fn):
    """
    Compute average gradients and loss for a mini-batch.
    net: the model.
    batch: list of (x, y_true) samples.
    loss_fn: callable loss function (e.g., cross_entropy, MSE).
    Returns:
        avg_grads: averaged gradients over the batch.
        avg_loss: average loss over the batch.
    """
    batch_grads = None
    batch_loss = 0.0

    for x, y_true in batch:
        y_pred = net.forward(x)
        batch_loss += loss_fn(y_true, y_pred)
        grads = net.backward(y_true, y_pred)

        # Accumulate grads.
        if batch_grads is None:
            batch_grads = grads
        else:
            for l in range(len(grads)):
                for key in grads[l]:
                    for i in range(len(grads[l][key])):
                        if isinstance(grads[l][key][i], list):
                            for j in range(len(grads[l][key][i])):
                                batch_grads[l][key][i][j] += grads[l][key][i][j]
                        else:
                            batch_grads[l][key][i] += grads[l][key][i]

    # Average grads.
    for l in range(len(batch_grads)):
        for key in batch_grads[l]:
            for i in range(len(batch_grads[l][key])):
                if isinstance(batch_grads[l][key][i], list):
                    for j in range(len(batch_grads[l][key][i])):
                        batch_grads[l][key][i][j] /= len(batch)
                else:
                    batch_grads[l][key][i] /= len(batch)

    return batch_grads, batch_loss / len(batch)

def set_training_mode(net, training=True):
    """
    Enable or disable training mode for layers that use it (e.g., Dropout).
    """
    for layer in net.layers:
        if hasattr(layer, "training"):
            layer.training = training

def init_weights(n_in, n_out, method="xavier"):
    """
    Initialize a list of weights given fan_in (n_in) and fan_out (n_out).
    Supports Xavier, He, or uniform random.
    """
    method = method.lower()
    if method == "xavier":
        limit = math.sqrt(6 / (n_in + n_out))
        return [random.uniform(-limit, limit) for _ in range(n_in)]
    elif method == "he":
        limit = math.sqrt(6 / n_in)
        return [random.uniform(-limit, limit) for _ in range(n_in)]
    else:
        return [random.uniform(-0.5, 0.5) for _ in range(n_in)]

def init_bias(method="zero"):
    """
    Initialize a bias term.
    method: "zero" (default) or "random".
    """
    if method == "zero":
        return 0.0
    elif method == "random":
        return random.uniform(-0.1, 0.1)
    else:
        raise ValueError(f"Unknown bias init method: {method}")
    
def win_num(L, P, F, S):
    """
    Compute number of sliding windows (output length).
    L = input length.
    P = padding.
    F = filter size.
    S = stride.
    """
    return ((L + 2*P - F) // S) + 1
