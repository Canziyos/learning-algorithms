# from sklearn.datasets import load_breast_cancer
# from .regression import LogisticRegression
# import numpy as np
# from .utils import mse

# data = load_breast_cancer()
# X = data.data               # all features.
# y = data.target.astype(float)

# print(data.feature_names)
# print(f"X shape: {X.shape}, y shape: {y.shape}.")

# # ---- 80/10/10 split (deterministic). ----
# n = X.shape[0]
# rng = np.random.default_rng(42)
# idx = rng.permutation(n)

# cut1 = int(0.8 * n)   # end of training set
# cut2 = int(0.9 * n)   # end of validation set

# tr, val, te = idx[:cut1], idx[cut1:cut2], idx[cut2:]
# Xtr, Xval, Xte = X[tr], X[val], X[te]
# ytr, yval, yte = y[tr], y[val], y[te]



# lrs = [0.001, 0.005, 0.1, 0.2, 0.5, 1]
# epochs = [100, 1000, 5000, 10000, 20000, 50000]


# for lr in lrs:
#     for epoch in epochs:
#         lg = LogisticRegression(lr=lr, epochs=epoch)
#         lg.fit(Xtr, ytr)
#         predicted = lg.predict_cls(Xval)
#         # Compare predictions to true labels
#         correct_predicted = np.sum(predicted == yval)

#         # Accuracy = correct / total
#         acc = correct_predicted/ len(yval)
# main_linear.py
import numpy as np

class LinearRegressionMulti:
    def __init__(self):
        self.w = None   # shape [d, k]
        self.b = None   # shape [k,]

    def fit(self, X, Y):
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        if X.ndim == 1: X = X[:, None]
        assert Y.ndim == 2, "Y must be 2-D shape (n, k) for multi-output."

        # Standardize X columns; center Y per column
        x_mean = X.mean(axis=0)
        x_std  = X.std(axis=0).copy()
        x_std[x_std == 0] = 1.0
        Xs = (X - x_mean) / x_std

        y_mean = Y.mean(axis=0)          # [k]
        Yc = Y - y_mean                  # [n, k]

        # Solve Xs @ B ≈ Yc via least squares (no intercept)
        # B has shape [d, k]
        B, *_ = np.linalg.lstsq(Xs, Yc, rcond=None)

        # Map back to original scale
        W = B / x_std[:, None]           # [d, k]
        b = y_mean - x_mean @ W          # [k]

        self.w = W
        self.b = b
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        if X.ndim == 1: X = X.reshape(1, -1)
        return X @ self.w + self.b       # [n, k]

import numpy as np
from .utils import load_data, split_data, normalize, mse_per_label

np.set_printoptions(precision=6, suppress=True)
np.random.seed(42)

# ---- Data: load & split (same as ANN) ----
X, y = load_data("neural_networks/data/maintenance.txt")
Xtr, ytr, Xva, yva, Xte, yte = split_data(X, y, 0.50, 0.25)
print(Xtr.shape, ytr.shape, Xva.shape, yva.shape, Xte.shape, yte.shape)

# ---- Normalize inputs using TRAIN stats (same as ANN) ----
Xtr_mean = np.mean(Xtr, axis=0)
Xtr_std  = np.std(Xtr, axis=0)
Xtr = normalize(Xtr, Xtr_mean, Xtr_std)
Xva = normalize(Xva, Xtr_mean, Xtr_std)
Xte = normalize(Xte, Xtr_mean, Xtr_std)

# ---- Multi-output linear least-squares with bias via X_aug ----
def fit_linear_lstsq(X, Y):
    """
    Solve for Theta in [X,1] @ Theta ≈ Y (least squares).
    X: [N, d], Y: [N, k]
    Returns W:[d,k], b:[k,]
    """
    N = X.shape[0]
    X_aug = np.hstack([X, np.ones((N, 1), dtype=X.dtype)])
    Theta, *_ = np.linalg.lstsq(X_aug, Y, rcond=None)  # [d+1, k]
    W = Theta[:-1, :]  # [d, k]
    b = Theta[-1, :]   # [k,]
    return W, b

def predict_linear(X, W, b):
    return X @ W + b

# ---- Fit on TRAIN ----
W, b = fit_linear_lstsq(Xtr, ytr)

# ---- Evaluate ----
# Train
ytr_hat = predict_linear(Xtr, W, b)
mse_c_tr, mse_t_tr = mse_per_label(ytr_hat, ytr)

# Val
yva_hat = predict_linear(Xva, W, b)
mse_c_va, mse_t_va = mse_per_label(yva_hat, yva)

# Test
yte_hat = predict_linear(Xte, W, b)
mse_c_te, mse_t_te = mse_per_label(yte_hat, yte)

# ---- Report ----
TH_COMP = 2.5e-08
TH_TURB = 6.0e-09

print("\n[Linear regression baseline]")
print(f"Train MSE  – Compressor: {mse_c_tr:.6e} | Turbine: {mse_t_tr:.6e}")
print(f"Val   MSE  – Compressor: {mse_c_va:.6e} | Turbine: {mse_t_va:.6e}")
print(f"Test  MSE  – Compressor: {mse_c_te:.6e} | Turbine: {mse_t_te:.6e}")
print(f"Targets    – Compressor ≤ {TH_COMP:.1e} | Turbine ≤ {TH_TURB:.1e}")

# Optional diagnostics
print("\n[Diag] Coefficient norms")
print(f"||W||_F: {np.linalg.norm(W):.6f} | ||b||_2: {np.linalg.norm(b):.6f}")
