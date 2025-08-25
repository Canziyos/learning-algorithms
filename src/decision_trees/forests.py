# bagging.py
import random
from classificatin_tree import ClassificationTree
from regression_tree import RegressionTree
from metrics import majority_class, accuracy, mse, mae, r2_score
from utils import resample_with_replacement


class Forest:
    def __init__(
        self,
        task="classification",   # "classification" or "regression".
        n_trees=10,
        max_depth=None,
        max_features=None,       # RF-style feature subsampling for BOTH trees.
        bootstrap=True,          # If False: all trees see full data (no OOB).
        random_state=1347,
        rng=None,                # optional random.Random instance.
    ):
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'.")
        if n_trees < 1:
            raise ValueError("n_trees must be >= 1.")
        self.task = task
        self.n_trees = int(n_trees)
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.rng = rng if rng is not None else random.Random(random_state)

        # Fitted state.
        self.forest = []             # list[ClassificationTree] or list[RegressionTree].
        self._oob_per_tree = []      # list[set[int]] per tree.
        self._train_features = None
        self._train_labels = None

    # Public API.
    def fit(self, features, labels):
        return self._train(features, labels)

    # Back-compat if you called train(...) elsewhere.
    def train(self, features, labels):
        return self._train(features, labels)

    # ----------------- training -----------------
    def _train(self, features, labels):
        self.forest = []
        self._oob_per_tree = []
        self._train_features = features
        self._train_labels = labels

        for _ in range(self.n_trees):
            if self.bootstrap:
                Xb, yb, oob_idx = resample_with_replacement(features, labels, rng=self.rng)
                oob_set = set(oob_idx)
            else:
                Xb, yb = features, labels
                oob_set = set()

            seed = self.rng.randrange(10**9)

            if self.task == "classification":
                tree = ClassificationTree(
                    max_depth=self.max_depth,
                    max_features=self.max_features,  # enables RF-style splits
                    random_state=seed,
                )
            else:  # regression
                tree = RegressionTree(
                    max_depth=self.max_depth,
                    max_features=self.max_features,  # RF-style split.
                    random_state=seed,
                )

            tree.fit(Xb, yb)
            self.forest.append(tree)
            self._oob_per_tree.append(oob_set)

        return self

    # ----------------- inference -----------------
    def predict_one(self, sample):
        if not self.forest:
            raise ValueError("Forest is not fitted; call fit(...) first.")
        if self.task == "classification":
            votes = [t.predict_one(sample) for t in self.forest]
            return majority_class(votes)
        else:
            preds = [t.predict_one(sample) for t in self.forest]
            return sum(preds) / len(preds)

    def predict_many(self, samples):
        if not self.forest:
            raise ValueError("Forest is not fitted; call fit(...) first.")
        return [self.predict_one(s) for s in samples]

    # Classification-only soft voting.
    def predict_proba_one(self, sample):
        if self.task != "classification":
            raise AttributeError("predict_proba_one is only available for classification.")
        if not self.forest:
            raise ValueError("Forest is not fitted; call fit(...) first.")
        all_classes, per_tree = set(), []
        for t in self.forest:
            p = t.predict_proba(sample)
            per_tree.append(p)
            all_classes.update(p.keys())
        n = len(self.forest)
        avg = {c: 0.0 for c in all_classes}
        for p in per_tree:
            for c in all_classes:
                avg[c] += p.get(c, 0.0)
        for c in avg:
            avg[c] /= n
        return avg

    def predict_many_proba(self, samples):
        if self.task != "classification":
            raise AttributeError("predict_many_proba is only available for classification.")
        return [self.predict_proba_one(s) for s in samples]

    # ----------------- OOB evaluation -----------------
    def oob_score(self):
        """Classification: returns accuracy.
            Regression: raises (use oob_scores)."""
        
        if self.task != "classification":
            raise AttributeError("oob_score is for classification. Use oob_scores() for regression.")
        if not self.forest or self._train_features is None:
            raise ValueError("Call fit(...) before oob_score().")
        if not self.bootstrap:
            raise ValueError("OOB score is undefined when bootstrap=False.")

        y_true, y_pred = [], []
        n = len(self._train_labels)
        for i in range(n):
            votes = []
            for t_idx, _ in enumerate(self.forest):
                if i in self._oob_per_tree[t_idx]:
                    sample = {f: self._train_features[f][i] for f in self._train_features}
                    votes.append(self.forest[t_idx].predict_one(sample))
            if votes:
                y_true.append(self._train_labels[i])
                y_pred.append(majority_class(votes))
        if not y_true:
            raise ValueError("No OOB predictions available. Try increasing n_trees.")
        return accuracy(y_true, y_pred)

    def oob_scores(self):
        """Regression: returns dict(mse, mae, r2, coverage).
        Classification: raises (use oob_score)."""

        if self.task != "regression":
            raise AttributeError("oob_scores is for regression. Use oob_score() for classification.")
        if not self.forest or self._train_features is None:
            raise ValueError("Call fit(...) before oob_scores().")
        if not self.bootstrap:
            raise ValueError("OOB scores are undefined when bootstrap=False.")

        y_true, y_pred = [], []
        n = len(self._train_labels)
        for i in range(n):
            preds = []
            for t_idx, _ in enumerate(self.forest):
                if i in self._oob_per_tree[t_idx]:
                    sample = {f: self._train_features[f][i] for f in self._train_features}
                    preds.append(self.forest[t_idx].predict_one(sample))
            if preds:
                y_true.append(self._train_labels[i])
                y_pred.append(sum(preds) / len(preds))
        if not y_true:
            raise ValueError("No OOB predictions available. Try increasing n_trees.")
        return {
            "mse": mse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "coverage": len(y_true) / n if n else 0.0,
        }

    def oob_coverage(self):
        if not self.forest or self._train_features is None:
            raise ValueError("Call fit(...) before oob_coverage().")
        if not self.bootstrap:
            return 0.0
        n = len(self._train_labels)
        covered = sum(1 for i in range(n) if any(i in s for s in self._oob_per_tree))
        return covered / n if n else 0.0


# Wrappers
class ClassificationForest(Forest):
    def __init__(self, **kwargs):
        super().__init__(task="classification", **kwargs)

class RegressionForest(Forest):
    def __init__(self, **kwargs):
        super().__init__(task="regression", **kwargs)
