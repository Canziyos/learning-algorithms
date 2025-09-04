from sklearn.datasets import load_breast_cancer
from .regression import LogisticRegression
import numpy as np
from .utils import mse
import matplotlib.pyplot as plt

data = load_breast_cancer()
X = data.data               # all features.
y = data.target.astype(float)

print(data.feature_names)
print(f"X shape: {X.shape}, y shape: {y.shape}.")

# ---- 80/10/10 split (deterministic). ----
n = X.shape[0]
rng = np.random.default_rng(42)
idx = rng.permutation(n)

cut1 = int(0.8 * n)   # end of training set
cut2 = int(0.9 * n)   # end of validation set

tr, val, te = idx[:cut1], idx[cut1:cut2], idx[cut2:]
Xtr, Xval, Xte = X[tr], X[val], X[te]
ytr, yval, yte = y[tr], y[val], y[te]



lrs = [0.001, 0.005, 0.1, 0.2, 0.5, 1]
epochs = [100, 1000, 5000, 10000, 20000, 50000]


for lr in lrs:
    for epoch in epochs:
        lg = LogisticRegression(lr=lr, epochs=epoch)
        lg.fit(Xtr, ytr)
        predicted = lg.predict_cls(Xval)
        # Compare predictions to true labels
        correct_predicted = np.sum(predicted == yval)

        # Accuracy = correct / total
        acc = correct_predicted/ len(yval)
