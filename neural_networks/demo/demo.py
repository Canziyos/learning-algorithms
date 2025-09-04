# demo_mnist.py
from conv2d import Conv2D
from downsample import Downsample
from flatten import Flatten
from dense import Dense
from model import Model
import utils, optim
from tqdm.auto import tqdm
from sklearn.datasets import fetch_openml
import numpy as np

# ---- checkpoint helpers ----
def save_checkpoint(net, path):
    state = {}
    for i, layer in enumerate(net.layers):
        if getattr(layer, "has_params", lambda: False)():
            p = layer.params()  # {"w": ..., "b": ...}
            state[f"{i}.type"] = layer.__class__.__name__
            for k, v in p.items():
                state[f"{i}.{k}"] = v
    np.savez(path, **state)


np.random.seed(42)

print("Loading MNIST via OpenML...")
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

# Normalize + reshape.
X = X.astype("float32") / 255.0
y = y.astype("int64")

x_train, x_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Reshape to (N, 1, 28, 28).
x_train = x_train.reshape(-1, 1, 28, 28)
x_test  = x_test.reshape(-1, 1, 28, 28)

# One-hot encode.
y_train_oh = np.eye(10, dtype=np.float32)[y_train]
y_test_oh  = np.eye(10, dtype=np.float32)[y_test]

print(f"Train set: {x_train.shape}, Test set: {x_test.shape}")

# 2. Define model.
conv1 = Conv2D(in_channels=1, n_kernels=8, height=3, width=3,
               step_size=1, padding=1, activation="relu")
pool1 = Downsample(pool_s=2, step_s=2, mode="max", dim=2)

conv2 = Conv2D(in_channels=8, n_kernels=16, height=3, width=3,
               step_size=1, padding=1, activation="relu")
pool2 = Downsample(pool_s=2, step_s=2, mode="max", dim=2)

# Compute Dense input size dynamically (Conv -> Pool -> Conv -> Pool)
out_h, out_w = utils.win_num_2d(28, 28, conv1.padding, conv1.height, conv1.width, conv1.step_s)
out_h = (out_h - pool1.pool_s) // pool1.step_s + 1
out_w = (out_w - pool1.pool_s) // pool1.step_s + 1

out_h, out_w = utils.win_num_2d(out_h, out_w, conv2.padding, conv2.height, conv2.width, conv2.step_s)
out_h = (out_h - pool2.pool_s) // pool2.step_s + 1
out_w = (out_w - pool2.pool_s) // pool2.step_s + 1

dense_in = conv2.n_kernels * out_h * out_w
dense = Dense(input_s=dense_in, output_s=10, activation="softmax")

net = Model([conv1, pool1, conv2, pool2, Flatten(), dense])

# 3. Training.
epochs = 3
batch_size = 32
lr = 0.005

for epoch in range(epochs):
    perm = np.random.permutation(len(x_train))
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_seen = 0

    prog_bar = tqdm(range(0, len(x_train), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
    for step, i in enumerate(prog_bar):
        batch_idx = perm[i:i+batch_size]
        xb, yb = x_train[batch_idx], y_train_oh[batch_idx]

        # Backward + update.
        grads, loss = utils.batch_backward(net, list(zip(xb, yb)), utils.cross_entropy)
        optim.update(net, grads, lr=lr, momentum=0.9, clip=None, weight_decay=1e-4)

        epoch_loss += loss

        # Training acc on this batch.
        preds = np.asarray(net.forward(xb))
        batch_correct = np.sum(np.argmax(preds, axis=1) == np.argmax(yb, axis=1))
        epoch_correct += batch_correct
        epoch_seen += len(xb)
        batch_acc = batch_correct / len(xb)

        # pb with metrics ---
        prog_bar.set_postfix({
            "loss": round(epoch_loss / (step + 1), 4),
            "batch_acc": round(batch_acc, 3),
            "lr": lr
        })

    # Evaluate on a subset(faster).
    correct = 0
    N_eval = 1000
    for x, y in zip(x_test[:N_eval], y_test_oh[:N_eval]):
        y_pred = net.forward(x)
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 2:
            y_pred = y_pred[0]
        if np.argmax(y_pred) == np.argmax(y):
            correct += 1
    acc = correct / N_eval

    train_acc = epoch_correct / epoch_seen
    print(f"Epoch {epoch+1}/{epochs}: loss={epoch_loss:.4f}, "
          f"train_acc={train_acc:.3f}, test_acc={acc:.3f}")

print("\nTraining complete.")

# save the trained weights.
ckpt_path = "mnist_cnn_v1.npz"
save_checkpoint(net, ckpt_path)
print(f"Saved checkpoint to {ckpt_path}.")
