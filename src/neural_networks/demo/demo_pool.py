import numpy as np
from pool import Pool

print("=== Pool1D: Max ===")
x = np.array([[1, 3, 2, 5, 4]])  # (n_filters=1, length=5)
pool1d = Pool(pool_s=2, step_s=2, mode="max", dim=1)
out1d = pool1d.forward(x)
print("Input:", x)
print("Output:", out1d)
grads = np.array([[1, 2]])  # gradient from next layer.
dx = pool1d.backward(grads)
print("Backward grads:", dx)

print("\n=== Pool1D: Avg ===")
pool1d_avg = Pool(pool_s=2, step_s=2, mode="avg", dim=1)
out1d_avg = pool1d_avg.forward(x)
print("Input:", x)
print("Output:", out1d_avg)
grads_avg = np.array([[1, 2]])
dx_avg = pool1d_avg.backward(grads_avg)
print("Backward grads:", dx_avg)

print("\n=== Pool2D: Max ===")
x2 = np.array([[[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]]])  # (n_filters=1, H=3, W=3)
pool2d = Pool(pool_s=2, step_s=1, mode="max", dim=2)
out2d = pool2d.forward(x2)
print("Input:\n", x2)
print("Output:\n", out2d)
grads2d = np.ones_like(out2d)
dx2d = pool2d.backward(grads2d)
print("Backward grads:\n", dx2d)

print("\n=== Pool2D: Avg ===")
pool2d_avg = Pool(pool_s=2, step_s=1, mode="avg", dim=2)
out2d_avg = pool2d_avg.forward(x2)
print("Input:\n", x2)
print("Output:\n", out2d_avg)
grads2d_avg = np.ones_like(out2d_avg)
dx2d_avg = pool2d_avg.backward(grads2d_avg)
print("Backward grads:\n", dx2d_avg)
