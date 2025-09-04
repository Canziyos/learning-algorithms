from node import Node
from utils import split_dataset, best_feature_split_regression, variance
from collections import Counter
import random
import math

class RegressionTree:
    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_gain=0.0,
        min_impurity_decrease=0.0,
        max_features=None,      # None => all features. "sqrt"/"log2"/int/float(0,1] => RF-style.
        random_state=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.root = None
        self._rng = random.Random(random_state)

    def fit(self, features, labels):
        # Cache feature count and resolve max_features once.
        self._d = len(features)
        self._resolved_max_features = self._resolve_max_features(self.max_features, self._d)
        self.root = self._build_tree(features, labels, depth=0)

    # Leaf predicts mean value.
    def _make_leaf(self, labels, depth):
        n = len(labels)
        value = sum(labels) / n if n > 0 else 0.0
        return Node(value=value, probs={}, depth=depth)

    # Map user option -> integer in [1, d].
    def _resolve_max_features(self, option, d):
        if d <= 0:
            return 0
        if option is None:
            return d
        if option == "sqrt":
            return max(1, int(math.sqrt(d)))
        if option == "log2":
            return max(1, int(math.log2(d)))
        if isinstance(option, int):
            return max(1, min(option, d))
        if isinstance(option, float):
            if option <= 0.0 or option > 1.0:
                raise ValueError("max_features as float must be in (0, 1].")
            return max(1, min(d, math.ceil(option * d)))
        raise ValueError("max_features must be None, 'sqrt', 'log2', int, or float in (0, 1].")

    # Choose a per-node random subset of features.
    def _choose_feature_subset(self, features_dict):
        names = list(features_dict.keys())
        k = self._resolved_max_features
        if k >= len(names):
            return names
        return self._rng.sample(names, k)

    def _build_tree(self, features, labels, depth):
        # Pure node for regression: zero variance.
        if variance(labels) == 0.0:
            return self._make_leaf(labels, depth)

        # Max depth.
        if self.max_depth is not None and depth >= self.max_depth:
            return self._make_leaf(labels, depth)

        # Min samples.
        if len(labels) < self.min_samples_split:
            return self._make_leaf(labels, depth)

        # ----- Random‑subset feature selection (RF bit). -----
        candidate_feats = self._choose_feature_subset(features)
        sub_features = {f: features[f] for f in candidate_feats}

        # Find best split via variance reduction among the subset.
        best_feat, best_thresh, best_gain = best_feature_split_regression(sub_features, labels)

        # No valid split.
        if best_feat is None:
            return self._make_leaf(labels, depth)

        # Gain thresholds.
        if best_gain < self.min_gain:
            return self._make_leaf(labels, depth)
        if best_gain < self.min_impurity_decrease:
            return self._make_leaf(labels, depth)

        # Split and recurse.
        (left_features, left_labels), (right_features, right_labels) = split_dataset(
            features, labels, best_feat, best_thresh
        )
        left_child = self._build_tree(left_features, left_labels, depth + 1)
        right_child = self._build_tree(right_features, right_labels, depth + 1)

        return Node(
            feature=best_feat,
            threshold=best_thresh,
            gain=best_gain,
            depth=depth,
            left=left_child,
            right=right_child,
        )

    def predict_one(self, sample):
        node = self.root
        while node.value is None:
            v = sample[node.feature]
            node = node.left if v <= node.threshold else node.right
        return node.value

    def predict_many(self, samples):
        return [self.predict_one(s) for s in samples]
