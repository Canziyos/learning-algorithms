# demo.py
# Modes:
# "cls" = classification tree
# "reg" = regression tree
# "bag_cls" = bagging forest (classification)
# "rf_cls" = random forest (classification)
# "bag_reg" = bagging forest (regression)
# "rf_reg" = random forest (regression)
TASK = "rf_reg"

from utils import load_dataset
from metrics import accuracy, mae, mse
from classificatin_tree import ClassificationTree
from regression_tree import RegressionTree
from forests import ClassificationForest, Forest


def run_classification(path, label_col, max_depth=3):
    features, labels, df, label_col = load_dataset(path, label_col)
    clf = ClassificationTree(max_depth=max_depth)
    clf.fit(features, labels)
    print("\n=== Classification Tree ===.")
    print("Root:", clf.root)
    feature_cols = [c for c in df.columns if c != label_col]
    sample = {col: df[col].iloc[0] for col in feature_cols}
    print("Prediction for first row:", clf.predict_one(sample))
    samples = [dict(zip(feature_cols, row)) for row in df[feature_cols].values]
    y_pred = clf.predict_many(samples)
    print("Accuracy on dataset:", accuracy(labels, y_pred), "\n")


def run_regression(path, label_col, max_depth=3):
    features, labels, df, label_col = load_dataset(path, label_col)
    rtf = RegressionTree(max_depth=max_depth)
    rtf.fit(features, labels)
    print("\n=== Regression Tree ===.")
    print("Root:", rtf.root)
    feature_cols = [c for c in df.columns if c != label_col]
    sample = {col: df[col].iloc[0] for col in feature_cols}
    print("Prediction for first row:", rtf.predict_one(sample))
    samples = [dict(zip(feature_cols, row)) for row in df[feature_cols].values]
    y_pred = rtf.predict_many(samples)
    print("MSE on dataset:", mse(labels, y_pred))
    print(f"MAE on dataset:", mae(labels, y_pred), "\n")


def run_bagging_classification(path, label_col, n_trees=25, max_depth=None, random_state=1347):
    features, labels, df, label_col = load_dataset(path, label_col)
    clf = ClassificationForest(
        n_trees=n_trees,
        max_depth=max_depth,
        max_features=None,   # classic bagging (no feature subspace at splits).
        bootstrap=True,
        random_state=random_state,
    )
    clf.fit(features, labels)
    print("\n=== Bagging Forest (Classification) ===.")
    feature_cols = [c for c in df.columns if c != label_col]
    sample = {col: df[col].iloc[0] for col in feature_cols}
    print("Prediction for first row:", clf.predict_one(sample))
    samples = [dict(zip(feature_cols, row)) for row in df[feature_cols].values]
    y_pred = clf.predict_many(samples)
    print("Accuracy on dataset (in-sample):", accuracy(labels, y_pred))
    try:
        oob = clf.oob_score()
        cov = clf.oob_coverage()
        print(f"OOB accuracy: {oob:.4f} (coverage: {cov:.3f}).\n")
    except ValueError as e:
        print("OOB not available:", e)


def run_random_forest_classification(path, label_col, n_trees=100, max_depth=None, max_features="sqrt", random_state=1347):
    features, labels, df, label_col = load_dataset(path, label_col)
    clf = ClassificationForest(
        n_trees=n_trees,
        max_depth=max_depth,
        max_features=max_features,  # "sqrt" | "log2" | int | float in (0,1]
        bootstrap=True,
        random_state=random_state,
    )
    clf.fit(features, labels)
    print("\n=== Random Forest (Classification) ===.")
    feature_cols = [c for c in df.columns if c != label_col]
    sample = {col: df[col].iloc[0] for col in feature_cols}
    print("Prediction for first row:", clf.predict_one(sample))
    samples = [dict(zip(feature_cols, row)) for row in df[feature_cols].values]
    y_pred = clf.predict_many(samples)
    print("Accuracy on dataset (in-sample):", accuracy(labels, y_pred))
    try:
        oob = clf.oob_score()
        cov = clf.oob_coverage()
        print(f"OOB accuracy: {oob:.4f} (coverage: {cov:.3f}).\n")
    except ValueError as e:
        print("OOB not available:", e)


def run_bagging_regression(path, label_col, n_trees=50, max_depth=None, random_state=1347):
    """Bagging regression forest: bootstrap + average predictions (no feature subspace)."""
    features, labels, df, label_col = load_dataset(path, label_col)
    rfb = Forest(
        task="regression",
        n_trees=n_trees,
        max_depth=max_depth,
        max_features=None,   # bagging (all features evaluated at splits)
        bootstrap=True,
        random_state=random_state,
    )
    rfb.fit(features, labels)
    print("\n=== Bagging Forest (Regression) ===.")
    feature_cols = [c for c in df.columns if c != label_col]
    sample = {col: df[col].iloc[0] for col in feature_cols}
    print("Prediction for first row:", rfb.predict_one(sample))
    samples = [dict(zip(feature_cols, row)) for row in df[feature_cols].values]
    y_pred = rfb.predict_many(samples)
    print("MSE on dataset (in-sample):", mse(labels, y_pred))
    print("MAE on dataset (in-sample):", mae(labels, y_pred))
    try:
        scores = rfb.oob_scores()
        print(f"OOB MSE: {scores['mse']:.4f}, OOB MAE: {scores['mae']:.4f}, OOB R2: {scores['r2']:.4f}, Coverage: {scores['coverage']:.3f}.\n")
    except ValueError as e:
        print("OOB not available:", e)


def run_random_forest_regression(path, label_col, n_trees=100, max_depth=None, max_features="sqrt", random_state=1347):
    """Random Forest Regressor: bagging + per-node random subset of features."""
    features, labels, df, label_col = load_dataset(path, label_col)
    rff = Forest(
        task="regression",
        n_trees=n_trees,
        max_depth=max_depth,
        max_features=max_features,   # e.g., "sqrt"
        bootstrap=True,
        random_state=random_state,
    )
    rff.fit(features, labels)
    print("\n=== Random Forest (Regression) ===.")
    feature_cols = [c for c in df.columns if c != label_col]
    sample = {col: df[col].iloc[0] for col in feature_cols}
    print("Prediction for first row:", rff.predict_one(sample))
    samples = [dict(zip(feature_cols, row)) for row in df[feature_cols].values]
    y_pred = rff.predict_many(samples)
    print("MSE on dataset (in-sample):", mse(labels, y_pred))
    print("MAE on dataset (in-sample):", mae(labels, y_pred))
    try:
        scores = rff.oob_scores()
        print(f"OOB MSE: {scores['mse']:.4f}, OOB MAE: {scores['mae']:.4f}, OOB R2: {scores['r2']:.4f}, Coverage: {scores['coverage']:.3f}.\n")
    except ValueError as e:
        print("OOB not available:", e)


def main():
    if TASK == "cls":
        path = "datasets/study_exam_cls.csv"
        label_col = "label"
        run_classification(path, label_col, max_depth=3)
    elif TASK == "reg":
        path = "datasets/study_exam_reg.csv"
        label_col = None
        run_regression(path, label_col, max_depth=3)
    elif TASK == "bag_cls":
        path = "datasets/study_exam_cls.csv"
        label_col = "label"
        run_bagging_classification(path, label_col, n_trees=25, max_depth=3)
    elif TASK == "rf_cls":
        path = "datasets/study_exam_cls.csv"
        label_col = "label"
        run_random_forest_classification(path, label_col, n_trees=100, max_depth=None, max_features="sqrt")
    elif TASK == "bag_reg":
        path = "datasets/study_exam_reg.csv"
        label_col = None
        run_bagging_regression(path, label_col, n_trees=50, max_depth=None)
    elif TASK == "rf_reg":
        path = "datasets/study_exam_reg.csv"
        label_col = None
        run_random_forest_regression(path, label_col, n_trees=100, max_depth=None, max_features="sqrt")
    else:
        raise ValueError("Unknown TASK. Use 'cls', 'reg', 'bag', 'rf', 'bag_reg', or 'rf_reg'.")


if __name__ == "__main__":
    main()
