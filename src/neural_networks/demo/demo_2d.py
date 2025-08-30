import os, numpy as np, matplotlib.pyplot as plt
from PIL import Image
from conv2d import Conv2D
from pool import Pool
from flatten import Flatten
from dense import Dense
from model import Model
import utils, optim

utils.DEBUG_LEVEL = 0   # keep it clean

# === 1. Load an image (original size) ===
img = Image.open(os.path.join(os.path.dirname(__file__), "test.png"))

# Pick scale based on bit depth
if img.mode in ["L", "RGB", "RGBA"]:
    scale = 255.0
elif img.mode in ["I;16", "I;16B"]:
    scale = 65535.0
else:
    scale = 1.0   # assume already normalized/float

arr = np.array(img) / scale

# Handle channels dynamically
if img.mode == "L":   # grayscale
    inputs = [arr.tolist()]     # [1][H][W]
elif img.mode == "RGB":
    inputs = [arr[:, :, c].tolist() for c in range(3)]   # [3][H][W]
elif img.mode == "RGBA":
    inputs = [arr[:, :, c].tolist() for c in range(4)]   # [4][H][W]
else:
    raise ValueError(f"Unsupported mode: {img.mode}")

print("Input shape:", len(inputs), "channels =>",
      len(inputs[0]), "x", len(inputs[0][0]))

# Dummy label (binary classification for testing)
y_true = [0]

# === 2. Define model (Conv2D -> Pool2D -> Flatten -> Dense) ===
conv = Conv2D(in_channels=len(inputs), n_kernels=4, height=3, width=3,
              step_size=1, padding=1, activation="relu")

pool = Pool(pool_s=2, step_s=2, mode="avg", dim=2)

# Compute Dense input size dynamically
out_h, out_w = utils.win_num_2d(len(inputs[0]), len(inputs[0][0]),
                                conv.padding, conv.height, conv.width,
                                conv.step_s)
out_h = (out_h - pool.pool_s) // pool.step_s + 1
out_w = (out_w - pool.pool_s) // pool.step_s + 1
dense_input_size = conv.n_kernels * out_h * out_w

dense = Dense(input_s=dense_input_size, n_neurons=1, activation="sigmoid")

net = Model([conv, pool, Flatten(), dense])

# === 3. Forward pass ===
out = net.forward(inputs)
print("\nNetwork output:", out)

# === NEW: visualize original + feature maps ===
feature_maps = conv.a_values  # [n_kernels][H][W]
n_maps = len(feature_maps)

fig, axes = plt.subplots(1, n_maps + 1, figsize=(3*(n_maps+1), 3))

# Show original image
if img.mode == "L":
    axes[0].imshow(arr, cmap="gray")
else:
    # Clip to 0..1 in case of float scaling
    axes[0].imshow(np.clip(arr, 0, 1))
axes[0].set_title("Original")
axes[0].axis("off")

# Show each feature map
for i, fmap in enumerate(feature_maps):
    axes[i+1].imshow(fmap, cmap="gray")
    axes[i+1].set_title(f"Kernel {i}")
    axes[i+1].axis("off")

plt.suptitle("Conv2D Feature Maps")
plt.show()

# === 4. Backward sanity check ===
grads = net.backward(y_true, out, loss="mse")
optim.update(net, grads, lr=0.01)

print("\nBackward => update done (no crash = success).")
