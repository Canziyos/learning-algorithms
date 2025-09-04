from collections import Counter
import math

def majority_class(labels):
    if not labels:
        raise ValueError("majority_class: 'labels' must be non-empty.")
    counts = Counter(labels)
    max_count = max(counts.values())
    candidates = [cls for cls, c in counts.items() if c == max_count]
    return min(candidates)  # smallest class label wins.

def accuracy(y_true, y_pred):
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        raise ValueError("accuracy: y_true and y_pred must be non-empty and equal length.")
    correct = 0
    for t, p in zip(y_true, y_pred):
        if isinstance(p, dict):
            if not p:
                pred_label = None
            else:
                max_p = max(p.values())
                candidates = [cls for cls, prob in p.items() if prob == max_p]
                pred_label = min(candidates)
        else:
            pred_label = p
        if t == pred_label:
            correct += 1
    return correct / len(y_true)

def mse(y_true, y_pred):
    n = len(y_true)
    if n == 0 or n != len(y_pred):
        raise ValueError("mse: y_true and y_pred must be non-empty and equal length.")
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n

def mae(y_true, y_pred):
    n = len(y_true)
    if n == 0 or n != len(y_pred):
        raise ValueError("mae: y_true and y_pred must be non-empty and equal length.")
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / n

def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return math.sqrt(mse(y_true, y_pred))

def r2_score(y_true, y_pred):
    """Coefficient of determination (R^2)."""
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("r2_score: y_true and y_pred must be non-empty and equal length.")
    y_mean = sum(y_true) / len(y_true)
    ss_tot = sum((t - y_mean) ** 2 for t in y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    # If variance is zero, define perfect fit when predictions equal true values.
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - (ss_res / ss_tot)
