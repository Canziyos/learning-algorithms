import numpy as np


def standardize(X):
    X = np.asarray(X, float)
    f_mean = np.mean(X, axis=0)
    f_std = np.std(X, axis=0)
    x_stand = (X-f_mean)/f_std
    return x_stand, f_mean, f_std

def mean_cost_calc(y_true, y_pred, eps=1e-15):

    X, y_true, y_pred = check_shape_dim(X, y_true, y_pred)

    y_pred = np.clip(y_pred, eps, 1 - eps)
    # per-sample loss.
    loss = y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred)
    cost = np.mean(loss)
    return -cost

def grad_calc(X, y_true, y_pred):
    X, y_true, y_pred = check_shape_dim(X, y_true, y_pred)
    errors = y_pred - y_true
    grad_w = (X.T @ errors) / X.shape[0]
    grad_b = np.mean(errors)

    return grad_w, grad_b

def check_shape_dim(X, y_true, y_pred):
    """
    Ensure X is (n_samples, n_features), y_true and y_pred are (n,), 
    and row counts match. Returns cleaned NumPy float arrays.
    """
    # Convert all to NumPy float arrays
    X = np.asarray(X, float)
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)

    # Check dimensions
    if X.ndim != 2:
        raise ValueError("X must be a 2D array (n_samples, n_features).")
    if y_true.ndim != 1:
        raise ValueError("y_true must be a 1D array (n_samples,).")
    if y_pred.ndim != 1:
        raise ValueError("y_pred must be a 1D array (n_samples,).")

    # Check row consistency
    if X.shape[0] != y_true.shape[0] or X.shape[0] != y_pred.shape[0]:
        raise ValueError("Number of rows in X, y_true, and y_pred must match.")

    return X, y_true, y_pred

def zscore_cols(X, eps=1e-12):
    """
    Z-score per column without ravel: Xs = (X - mean) / std.
    Returns: Xs (same shape as X), mu (1D), sigma (1D, safe).
    """
    X = np.asarray(X, float)
    mu = X.mean(axis=0)                       # 1D (m,)
    sigma = X.std(axis=0, ddof=0)             # 1D (m,)
    sigma_safe = sigma.copy()
    sigma_safe[sigma_safe < eps] = 1.0        # avoid divide-by-zero
    Xs = (X - mu) / sigma_safe                # broadcasts
    return Xs, mu, sigma_safe


def normal_eq_no_intercept(X, yc):
    """
    Solve (X^T X) beta = X^T yc without an intercept.
    Expects yc centered (mean 0). Returns a 1-D coefficient vector.
    """
    X = np.asarray(X, float)
    yc = np.asarray(yc, float)
    assert yc.ndim == 1 and X.shape[0] == yc.shape[0], "Shape mismatch."
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ yc)   # or: np.linalg.lstsq(X, yc, rcond=None)[0]
    assert beta.ndim == 1, "beta must be 1-D."
    return beta  # 1-D (m,).

def grad_no_intercept(X, yc, beta):
    """
    Gradient of (1/n)||X beta - yc||^2 with no intercept.
    Expects yc centered and beta 1-D.
    """
    X = np.asarray(X, float)
    yc = np.asarray(yc, float)
    beta = np.asarray(beta, float)
    assert yc.ndim == 1 and beta.ndim == 1, "yc and beta must be 1-D."
    n, m = X.shape
    assert yc.shape[0] == n and beta.shape[0] == m, "Shape mismatch."
    resid = X @ beta - yc
    return (2.0 / n) * (X.T @ resid)

def intercept_from_means(x_mean, y_mean, coef):
    """
    Recover intercept on the original scale once coef is on the original scale.
    """
    x_mean = np.asarray(x_mean, float)
    coef = np.asarray(coef, float)
    assert x_mean.ndim == 1 and coef.ndim == 1 and x_mean.shape[0] == coef.shape[0], "Shape mismatch."
    return float(y_mean - x_mean @ coef)


def mse(y_pred, y_true):
    y_pred = np.asarray(y_pred, float).ravel()
    y_true = np.asarray(y_true, float).ravel()
    if y_pred.size != y_true.size:
        raise ValueError("Size mismatch.")
    return np.mean((y_pred - y_true) ** 2)



def make_model_fn(model):
    """
    Returns a vectorized function f(X) -> y_hat using model.b and model.w.
    X can be shape (n, m) or (m,).
    """
    w = np.asarray(model.w, float)
    b = float(model.b)
    def f(X):
        X = np.asarray(X, float)
        if X.ndim == 1:
            return float(X @ w + b)
        return X @ w + b
    return f

def make_named_model_fn(model, feature_names):
    """
    Returns a callable f(**features) -> y_hat using named keyword args in the given order.
    Example: f(age=..., sex=..., bmi=..., ..., s6=...)
    """
    names = list(feature_names)
    w = np.asarray(model.w, float)
    b = float(model.b)
    def f(**kw):
        x = np.array([kw[name] for name in names], float)
        return float(x @ w + b)
    return f

def linear_eq_string(model, feature_names):
    terms = [f"{model.b:.6f}"]
    for name, coef in zip(feature_names, model.w):
        terms.append(f"{coef:.6f}*{name}")
    return "y = " + " + ".join(terms) + "."
