from collections import Counter

def majority_class(labels):
    counts = Counter(labels)
    max_count = max(counts.values())
    candidates = [cls for cls, c in counts.items() if c == max_count]
    return min(candidates)   # smallest class label wins

def accuracy(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Invalid input: y_true and y_pred must be non-empty and equal length.")
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
