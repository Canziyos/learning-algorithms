from layer_base import Layer
from utils import dprint
import numpy as np

class Model:
    def __init__(self, layers):
        for l in layers:
            if not isinstance(l, Layer):
                raise TypeError("All layers must inherit from Layer base class.")
        self.layers = layers
        self.n_layers = len(layers)

    def forward(self, inputs):
        x = inputs
        self.layer_inputs = []
        self.layer_outputs = []
        dprint(2, "\n[Model Forward]")
        for idx, layer in enumerate(self.layers):
            self.layer_inputs.append(x)
            x = layer.forward(x)
            self.layer_outputs.append(x)

            # 👇 Shape-aware debug
            arr = np.array(x, dtype=object) if hasattr(x, "__len__") else None
            shape_info = f"shape {arr.shape}" if arr is not None else "scalar"
            dprint(2, f" Layer {idx}: {layer.describe()} → {shape_info}")

        return x

    def backward(self, y_true, y_pred, loss="mse"):
        grads = [None] * self.n_layers

        # 1. Initial gradient from loss
        if loss == "cross_entropy" and hasattr(self.layers[-1], "activation") and \
           self.layers[-1].activation.__name__ == "softmax":
            grad_out = [yp - yt for yp, yt in zip(y_pred, y_true)]
        else:
            grad_out = [(2/len(y_true)) * (yp - yt) for yp, yt in zip(y_pred, y_true)]

        dprint(2, "\n[Model Backward]")
        dprint(2, f" Initial grad_out from loss: {grad_out}")

        # 2. Backward pass through layers
        for l in reversed(range(self.n_layers)):
            layer = self.layers[l]
            dprint(2, f"\n Backward through Layer {l}: {layer.describe()}")

            if layer.has_params():
                param_grads, grad_out = layer.backward(grad_out)
                grads[l] = param_grads
                dprint(2, f"   Param grads: {param_grads}")
            else:
                grad_out = layer.backward(grad_out)
                grads[l] = {"w": [], "b": []}  # placeholder
                dprint(2, f"   Non-param layer grad_out: {grad_out}")

            dprint(2, f"   grad_out passed upstream: {grad_out}")

        return grads

    def summary(self):
        print("\nModel Summary:")
        print(f"  Total layers: {self.n_layers}")
        for i, layer in enumerate(self.layers):
            print(f"  Layer {i+1}: {layer.describe()}")
