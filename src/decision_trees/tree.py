from collections import Counter
from .node import Node
from .utils import split_dataset, best_feature_split


def build_tree(features, labels, depth=0, max_depth=None,
               min_samples_split=2, min_gain=0.0):
    if len(set(labels)) == 1:
        return Node(value=labels[0], depth=depth)

    if max_depth is not None and depth >= max_depth:
        majority = Counter(labels).most_common(1)[0][0]
        return Node(value=majority, depth=depth)

    if len(labels) < min_samples_split:
        majority = Counter(labels).most_common(1)[0][0]
        return Node(value=majority, depth=depth)

    best_feat, best_thresh, best_gain = best_feature_split(features, labels)
    if best_gain == 0 or best_gain < min_gain:
        majority = Counter(labels).most_common(1)[0][0]
        return Node(value=majority, depth=depth)

    (left_features, left_labels), (right_features, right_labels) = split_dataset(
        features, labels, best_feat, best_thresh
    )

    left_child = build_tree(left_features, left_labels, depth+1,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_gain=min_gain)

    right_child = build_tree(right_features, right_labels, depth+1,
                             max_depth=max_depth,
                             min_samples_split=min_samples_split,
                             min_gain=min_gain)

    return Node(feature=best_feat, threshold=best_thresh,
                gain=best_gain, depth=depth,
                left=left_child, right=right_child)


def predict_one(tree, sample):
    node = tree
    while node.value is None:
        v = sample[node.feature]
        node = node.left if v <= node.threshold else node.right
    return node.value


def predict_many(tree, samples):
    return [predict_one(tree, sample) for sample in samples]


def accuracy(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
        raise ValueError("you fucked up")
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)
