from collections import Counter
from node import Node
from utils import split_dataset, best_feature_split

class ClassificationTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_gain=0.0, min_impurity_decrease=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.min_impurity_decrease = min_impurity_decrease
        self.root = None

    def fit(self, features, labels):
        self.root = self._build_tree(features, labels, depth=0)

    # Helper to build a leaf with class probs and deterministic tie-break.
    def _make_leaf(self, labels, depth):
        counts = Counter(labels)
        total = len(labels)
        probs = {cls: c / total for cls, c in counts.items()}
        max_count = max(counts.values())
        tie_classes = [cls for cls, c in counts.items() if c == max_count]
        majority = sorted(tie_classes)[0]
        return Node(value=majority, probs=probs, depth=depth)

    def _build_tree(self, features, labels, depth):
        # Pure node.
        if len(set(labels)) == 1:
            return self._make_leaf(labels, depth)

        # Max depth.
        if self.max_depth is not None and depth >= self.max_depth:
            return self._make_leaf(labels, depth)

        # Min samples.
        if len(labels) < self.min_samples_split:
            return self._make_leaf(labels, depth)

        # Find best split.
        best_feat, best_thresh, best_gain = best_feature_split(features, labels)

        # No valid split.
        if best_feat is None:
            return self._make_leaf(labels, depth)

        # Gain thresholds (relative + absolute).
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

        return Node(feature=best_feat, threshold=best_thresh, gain=best_gain,
                    depth=depth, left=left_child, right=right_child)

    def predict_one(self, sample):
        node = self.root
        while node.value is None:
            v = sample[node.feature]
            node = node.left if v <= node.threshold else node.right
        return node.value

    def predict_many(self, samples):
        return [self.predict_one(s) for s in samples]

    def predict_proba(self, sample):
        node = self.root
        while node.value is None:
            v = sample[node.feature]
            node = node.left if v <= node.threshold else node.right
        return node.probs

    def predict_many_proba(self, samples):
        return [self.predict_proba(s) for s in samples]

    def __repr__(self):
        if self.root is None:
            return "ClassificationTree(untrained)."
        lines = []
        def walk(n, pref=""):
            if n.value is not None:
                lines.append(f"{pref}Leaf(value={n.value}, probs={n.probs}, depth={n.depth}).")
            else:
                lines.append(f"{pref}Node(feature={n.feature}, thresh={n.threshold}, gain={n.gain:.4f}, depth={n.depth}).")
                walk(n.left,  pref + "  L- ")
                walk(n.right, pref + "  R- ")
        walk(self.root)
        return "\n".join(lines)
