import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def MSE(y_true, y_pred):
    """
    Mean squared error loss.
    Supports vectors or batches (arrays with the same shape).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("MSE: y_true and y_pred must be non-empty and equal shape.")
    
    return np.mean(np.mean((y_true - y_pred) ** 2, axis=tuple(range(1, y_true.ndim))))



def cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Cross-entropy loss for multi-class classification with encoded target vector..
    Supports vectors or batches (same shape).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("cross_entropy: y_true and y_pred must be the same shape.")

    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))


def batch_backward(net, batch, loss_fn):
    """
    Compute gradients and loss for a mini-batch.

    This runs a single vectorized forward/backward.
     Otherwise, it falls back to a per-sample loop.

    Returns:
        grad : averaged gradients over the batch
        avg_loss : average loss over the batch
    """

    loss_name = getattr(loss_fn, "__name__", "")

    # Fast path: will try vectorized batch (one forward/backward).
    try:
        X = np.stack([np.asarray(x) for x, _ in batch], axis=0)
        Y = np.stack([np.asarray(y) for _, y in batch], axis=0)

        Y = np.asarray(Y)
        Y_pred = np.asarray(net.forward(X))

        avg_loss = loss_fn(Y, Y_pred)

        grads = net.backward(
            Y, Y_pred,
            loss="cross_entropy" if loss_name == "cross_entropy" else "mse"
        )
        # grads are already averaged over batch because Model.backward seeds with 1/B
        return grads, avg_loss

    except Exception:
        # Safe fallback: per-sample loop + manual averaging.
        batch_grads = None
        batch_loss = 0.0
        N = len(batch)

        for x, y_true in batch:
            y_pred = net.forward(x)

            y_true = np.asarray(y_true)
            y_pred = np.asarray(y_pred)

            batch_loss += loss_fn(y_true, y_pred)

            grads = net.backward(
                y_true, y_pred,
                loss="cross_entropy" if loss_name == "cross_entropy" else "mse"
            )

            if batch_grads is None:
                batch_grads = grads
            else:
                for l in range(len(grads)):
                    if not grads[l]:
                        continue
                    for key, val in grads[l].items():
                        if isinstance(val, np.ndarray):
                            batch_grads[l][key] += val

        # average after the loop.
        for l in range(len(batch_grads)):
            if not batch_grads[l]:
                continue
            for key, val in batch_grads[l].items():
                if isinstance(val, np.ndarray):
                    batch_grads[l][key] /= N

        return batch_grads, batch_loss / N


def init_weights(n_out, n_in, method="xavier"):
    """
    Initialize weights with shape (n_out, n_in).
    Methods: "xavier", "he", or uniform fallback.
    """
    method = method.lower()
    if method == "xavier":
        limit = np.sqrt(6 / (n_out + n_in))
        return np.random.uniform(-limit, +limit, (n_out, n_in))
    elif method == "he":
        limit = np.sqrt(6 / n_in)
        return np.random.uniform(-limit, +limit, (n_out, n_in))
    else:
        return np.random.uniform(-0.5, 0.5, (n_out, n_in))


def init_bias(n_out, method="zero"):
    """
    Initialize biases with shape (n_out,).
    """
    if method == "zero":
        return np.zeros(n_out)
    elif method == "random":
        return np.random.uniform(-0.1, 0.1, n_out)
    else:
        raise ValueError(f"Unknown bias init method: {method}")


def win_slide_1d(seq, win_s, step):
    seq = np.asarray(seq)
    windows = sliding_window_view(seq, win_s)[::step]
    return windows


def win_slide_2d(seq, win_s, step):
    seq = np.asarray(seq)
    windows = sliding_window_view(seq, (win_s, win_s))[::step, ::step]
    return windows.reshape(-1, win_s, win_s)


def win_num_2d(H, W, P, K_h, K_w, S):
    """
    Compute output height and width for 2D convolution.
    """
    out_H = ((H + 2*P - K_h) // S) + 1
    out_W = ((W + 2*P - K_w) // S) + 1
    return out_H, out_W


def win_num_1d(L, P, F, S):
    """
    Compute number of sliding windows (output length).
    """
    return ((L + 2*P - F) // S) + 1
