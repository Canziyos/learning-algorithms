from .utils import normal_eq_no_intercept, grad_no_intercept, zscore_cols
import numpy as np


class Regression:
    def fit(self, X, y):
        raise NotImplementedError("Must be implemented in subclass.")

    def predict(self, X):
        raise NotImplementedError("Must be implemented in subclass.")


class LinearRegression(Regression):
    def __init__(self, lr=0.01, epochs=1000):
        self.w = None
        self.b = None
        self.lr = lr
        self.epochs = epochs
        self.beta = None

    def fit(self, X, y, method="norm"):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        assert y.ndim == 1, "y must be 1-D shape (n,)."
        if X.ndim == 1:
            X = X[:, None]

        # 1) Standardize X, center y.
        Xs, x_mean, x_std = zscore_cols(X)   # (X - mean) / std per column.
        y_mean = float(y.mean())
        yc = y - y_mean                      # centered target (no intercept in solver).

        # 2) Train on (Xs, yc) with NO intercept.
        if method == "norm":
            beta_std = normal_eq_no_intercept(Xs, yc)
        elif method == "grad":
            beta_std = np.zeros(Xs.shape[1], dtype=float)
            for _ in range(self.epochs):
                g = grad_no_intercept(Xs, yc, beta_std)
                beta_std -= self.lr * g
        else:
            raise ValueError("Unsupported method.")

        # 3) Map back to original feature scale + recover intercept.
        coef = beta_std / x_std
        b = y_mean - x_mean @ coef

        # Store.
        self.w = coef
        self.b = float(b)
        self.beta = np.concatenate(([self.b], self.w))
        return self



    def predict(self, x):
        if self.w is None or self.b is None:
            raise ValueError("Model is not fitted yet. Call fit() before predict().")
        x = np.asarray(x, float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x @ self.w + self.b


from src.neural_networks.activations import sigmoid
from .utils import mean_cost_calc, grad_calc

class LogisticRegression(Regression):
    def __init__(self, lr = 0.01, epochs=2000):
        self.w = None
        self.b = None
        self.epochs = epochs
        self.lr = lr
        self.beta = None

    def fit(self, X, y):
        # Initialize parameters
        self.w = np.zeros(X.shape[1])
        self.b = 0.0

        # Track cost (optional)
        costs = []

        for epoch in range(self.epochs):
            # Linear scores
            z_i = X @ self.w + self.b

            # Predictions
            y_pred = sigmoid(z_i)

            # Compute cost
            mean_cost = mean_cost_calc(y, y_pred)
            costs.append(mean_cost)

            # Compute gradients
            grad_w, grad_b = grad_calc(X, y, y_pred)

            # Update parameters
            self.w = self.w - self.lr * grad_w
            self.b = self.b - self.lr * grad_b

            # Print cost occasionally (every 50 epochs)
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - Cost: {mean_cost:.6f}")

        # Store parameters in beta for consistency
        self.beta = np.concatenate(([self.b], self.w))

        return self


    def predict_prob(self, X):
        z = X@self.w + self.b
        y_pred = sigmoid(z)
        return y_pred
        
    def predict_cls(self, X, threshold=0.5):
        """
        Predict class labels (0 or 1) from input X using given threshold.
        """
        # get probabilities from predict_prob.
        probs = self.predict_prob(X)

        # apply threshold: >= threshold => 1, else => 0
        return (probs >= threshold).astype(int)
