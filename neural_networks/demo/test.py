# test_mnist.py
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

from conv2d import Conv2D
from downsample import Downsample
from flatten import Flatten
from dense import Dense
from model import Model

ckpt_path = Path(__file__).resolve().parent / "mnist_cnn_v1.npz"

batch_eval = 256
show_grid = False           # set true to visualize 25 random predictions.
confusion = False           # set True for confusion matrix and report.

def build_model():
    # The same architecture used during training.
    conv1 = Conv2D(in_channels=1, n_kernels=8, height=3, width=3,
                   step_size=1, padding=1, activation="relu")
    pool1 = Downsample(pool_s=2, step_s=2, mode="max", dim=2)
    conv2 = Conv2D(in_channels=8, n_kernels=16, height=3, width=3,
                   step_size=1, padding=1, activation="relu")
    pool2 = Downsample(pool_s=2, step_s=2, mode="max", dim=2)
    dense_in = conv2.n_kernels * 7 * 7  # After 2x (3x3 stride1 pad1) + 2x 2x2 max pool.
    dense = Dense(input_s=dense_in, output_s=10, activation="softmax")
    return Model([conv1, pool1, conv2, pool2, Flatten(), dense])

def load_checkpoint(net, path):
    data = np.load(path, allow_pickle=False)
    for i, layer in enumerate(net.layers):
        if getattr(layer, "has_params", lambda: False)():
            # check stored type.
            key_type = f"{i}.type"
            if key_type in data:
                saved_type = str(data[key_type])
                if saved_type != layer.__class__.__name__:
                    raise ValueError(f"Layer {i} type mismatch: file={saved_type}, code={layer.__class__.__name__}.")
            # Assign params.
            params = layer.params()  # keys: "w", "b".
            for k in params.keys():
                arr = data[f"{i}.{k}"]
                if params[k].shape != arr.shape:
                    raise ValueError(f"Layer {i} param {k} shape mismatch: {params[k].shape} vs {arr.shape}.")
                if hasattr(layer, "w") and k == "w":
                    layer.w = arr
                if hasattr(layer, "b") and k == "b":
                    layer.b = arr
                if hasattr(layer, "kernels") and k == "w":
                    layer.kernels = arr
                if hasattr(layer, "biases") and k == "b":
                    layer.biases = arr

def load_mnist_test():
    # Fetch MNIST and slice the test split.
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X.astype("float32") / 255.0
    y = y.astype("int64")
    x_test = X[60000:].reshape(-1, 1, 28, 28)
    y_test = y[60000:]
    return x_test, y_test

def eval_full(net, x_test, y_test, batch=batch_eval):
    preds = np.empty((len(x_test),), dtype=np.int64)
    pbar = tqdm(range(0, len(x_test), batch), desc="Eval", leave=False)
    for i in pbar:
        xb = x_test[i:i+batch]
        logits = np.asarray(net.forward(xb))  # shape (B, 10)
        # Softmax checks.
        if not np.isfinite(logits).all():
            raise RuntimeError("Non-finite logits detected during evaluation.")
        # If last layer already applies softmax, sums should be ~1 along axis=-1.
        row_sums = logits.sum(axis=-1)
        if not np.allclose(row_sums, 1.0, atol=1e-4):
            # If Dense(softmax) kept, these should be ~1. If not, comment this out.
            raise RuntimeError("Softmax outputs do not sum to 1. Check activations/softmax.")
        preds[i:i+len(xb)] = np.argmax(logits, axis=1)
    acc = accuracy_score(y_test, preds)
    return preds, acc

def main():
    # Check checkpoint file.
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}.")

    # Load test data.
    x_test, y_test = load_mnist_test()
    print(f"Test set: {x_test.shape}.")

    # Build model and load weights.
    net = build_model()
    load_checkpoint(net, ckpt_path)
    print("Loaded checkpoint into model.")

    # Evaluate on the full test set.
    y_pred, acc = eval_full(net, x_test, y_test, batch=batch_eval)
    print(f"Full test accuracy (10,000): {acc:.4f}.")

    # Grid of predictions (optional, change flag above).
    if show_grid:
        np.random.seed(0)
        sel = np.random.choice(len(x_test), size=25, replace=False)
        fig, axes = plt.subplots(5, 5, figsize=(7, 7))
        for ax, i in zip(axes.ravel(), sel):
            ax.imshow(x_test[i, 0], cmap="gray")
            t, p = int(y_test[i]), int(y_pred[i])
            ax.set_title(f"T{t}/P{p}{'OK' if p==t else ' No'}", fontsize=8)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    # Cnfusion matrix and per-class report (Optional: flag).
    if confusion:
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion matrix (rows=true, cols=pred).")
        print(cm)
        print("\nPer-class report.")
        print(classification_report(y_test, y_pred, digits=4))

    mis = np.flatnonzero(y_pred != y_test)
    print(f"Misclassified: {mis.size}/{len(y_test)}.")
    print("First 20 mistakes (idx: true -> pred):")
    for i in mis[:20]:
        print(f"{i}: {int(y_test[i])} -> {int(y_pred[i])}.")

if __name__ == "__main__":
    main()
