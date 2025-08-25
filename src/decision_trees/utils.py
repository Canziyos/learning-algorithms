import pandas as pd
from collections import Counter

def gini(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    impurity = 1.0
    inv_n = 1.0 / n
    for c in counts.values():
        p = c * inv_n
        impurity -= p * p
    return impurity


def split_data(feature, labels, threshold):
    left_labels, right_labels = [], []
    for feat_val, lbl in zip(feature, labels):
        if feat_val <= threshold:
            left_labels.append(lbl)
        else:
            right_labels.append(lbl)
    return left_labels, right_labels


def split_gain(feature, labels, threshold):
    parent_imp = gini(labels)
    left_labels, right_labels = split_data(feature, labels, threshold)
    n_total = len(labels)

    if len(left_labels) == 0 or len(right_labels) == 0:
        return 0

    weighted_imp = (len(left_labels) / n_total) * gini(left_labels) + \
                   (len(right_labels) / n_total) * gini(right_labels)
    return parent_imp - weighted_imp


def split_dataset(features, labels, feature_name, threshold):
    left_features = {f: [] for f in features}
    right_features = {f: [] for f in features}
    left_labels, right_labels = [], []

    for i in range(len(labels)):
        value = features[feature_name][i]
        if value <= threshold:
            left_labels.append(labels[i])
            for f in features:
                left_features[f].append(features[f][i])
        else:
            right_labels.append(labels[i])
            for f in features:
                right_features[f].append(features[f][i])

    return (left_features, left_labels), (right_features, right_labels)

def best_split(feature, labels):
    best_gain, best_threshold = 0, None
    sorted_vals = sorted(set(feature))

    # no possible split if all feature values identical.
    if len(sorted_vals) < 2:
        return 0, None

    for i in range(len(sorted_vals) - 1):
        threshold = (sorted_vals[i] + sorted_vals[i+1]) / 2
        gain = split_gain(feature, labels, threshold)
        if gain > best_gain:
            best_gain, best_threshold = gain, threshold

    return best_gain, best_threshold

def best_feature_split(features, labels):
    best_gain = -1.0
    best_feature = None
    best_thresh = None
    best_feat_idx = None

    for idx, (feat_name, feat_values) in enumerate(features.items()):
        gain, thresh = best_split(feat_values, labels)
        if thresh is None:
            continue

        better = (
            gain > best_gain
            or (gain == best_gain and (best_feat_idx is None or idx < best_feat_idx))
            or (gain == best_gain and idx == best_feat_idx and (best_thresh is None or thresh < best_thresh))
        )

        if better:
            best_gain = gain
            best_feature = feat_name
            best_thresh = thresh
            best_feat_idx = idx

    if best_feature is None:
        return None, None, 0

    return best_feature, best_thresh, best_gain


def load_dataset(path, label):
    data = pd.read_csv(path)
    labels = data[label].tolist()
    features = {col: data[col].tolist() for col in data.columns if col != label}
    return features, labels, data, label
