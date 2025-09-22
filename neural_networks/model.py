# model.py
from layer_base import Layer
import numpy as np

class Model:
    def __init__(self, layers, shape_debug=False):
        for l in layers:
            if not isinstance(l, Layer):
                raise TypeError("All layers must inherit from Layer base class.")
        self.layers = layers
        self.n_layers = len(layers)
        self.shape_debug = shape_debug

    def forward(self, inputs):
        """
        Passes inputs through the stack.
        Accepts single sample or batched tensors; layers decide shape handling.
        """
        x = np.asarray(inputs)
        if x.ndim == 1:
            x = x[None, :]  # promote to batch

        self.layer_inputs = []
        self.layer_outputs = []
        for li, layer in enumerate(self.layers):
            self.layer_inputs.append(x)
            x = layer.forward(x)
            self.layer_outputs.append(x)

            if self.shape_debug:
                print(f"[Layer {li+1} {layer.describe()}] in={self.layer_inputs[-1].shape}, out={x.shape}")

            # enforce batch dimension stays intact.
            assert x.shape[0] == self.layer_inputs[-1].shape[0], \
                f"Batch dim mismatch in {layer.describe()}: {self.layer_inputs[-1].shape} -> {x.shape}"

        return x


    def backward(self, y_true, y_pred, loss="mse"):
        """
        Seeds gradient from loss and backprops through layers.
        Supports (K,) or (B,K) shapes for y_true/y_pred.
        Returns: list of per-layer param grads ({} for non-param layers).
        """
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)

        # Normalize to (B, K)
        if y_pred.ndim == 1: y_pred = y_pred[None, :]
        if y_true.ndim == 1: y_true = y_true[None, :]
        B = y_true.shape[0]
        K = y_true.shape[1]  # per-sample dimension

        # Determine last activation (if any),
        last = self.layers[-1]
        last_act = getattr(last, "activation", None)
        last_act_name = getattr(last_act, "__name__", "") if last_act else ""

        # Seed gradient from loss.
        if loss == "cross_entropy" and last_act_name == "softmax":
            # classic simplification for softmax+CE.
            grad_out = (y_pred - y_true) / B
        else:
            # MSE-style seed, consistent with per-sample mean then batch mean.
            grad_out = (2.0 / (B * K)) * (y_pred - y_true)

        grads = [None] * self.n_layers

        # Backprop.
        for li in reversed(range(self.n_layers)):
            layer = self.layers[li]
            if layer.has_params():
                param_grads, grad_out = layer.backward(grad_out)
                grads[li] = param_grads
            else:
                grad_out = layer.backward(grad_out)
                grads[li] = {}  # no params to update

        return grads

    def summary(self):
        print("\nModel Summary:")
        print(f"- Total layers: {self.n_layers}")
        for i, layer in enumerate(self.layers):
            print(f"- Layer {i+1}: {layer.describe()}")
