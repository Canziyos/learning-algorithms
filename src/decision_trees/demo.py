# demo.py

TASK = "cls"  # "cls" for classification, or "reg" for regression.

from utils import load_dataset
from metrics import accuracy, mae, mse

# Models.
from classificatin_tree import ClassificationTree
from regression_tree import RegressionTree

def run_classification(path, label_col, max_depth=3):
    features, labels, df, label_col = load_dataset(path, label_col)
    clf = ClassificationTree(max_depth=max_depth)
    clf.fit(features, labels)
    print("=== Classification Tree ===.")
    print(clf)
    print("Root:", clf.root)
    feature_cols = [c for c in df.columns if c != label_col]
    sample = {col: df[col].iloc[0] for col in feature_cols}
    print("Prediction for first row:", clf.predict_one(sample))
    samples = [dict(zip(feature_cols, row)) for row in df[feature_cols].values]
    y_pred = clf.predict_many(samples)
    print("Accuracy on dataset:", accuracy(labels, y_pred))

def run_regression(path, label_col, max_depth=3):
    features, labels, df, label_col = load_dataset(path, label_col)
    rtf = RegressionTree(max_depth=max_depth)
    rtf.fit(features, labels)
    print("=== Regression Tree ===.")
    print(rtf)
    print("Root:", rtf.root)
    feature_cols = [c for c in df.columns if c != label_col]
    sample = {col: df[col].iloc[0] for col in feature_cols}
    print("Prediction for first row:", rtf.predict_one(sample))
    samples = [dict(zip(feature_cols, row)) for row in df[feature_cols].values]
    y_pred = rtf.predict_many(samples)
    print("MSE on dataset:", mse(labels, y_pred))
    print("MAE on dataset:", mae(labels, y_pred))

def main():
    if TASK == "cls":
        path = "datasets/study_exam_cls.csv"
        label_col = "label"
        run_classification(path, label_col, max_depth=3)
    else:
        path = "datasets/study_exam_reg.csv"
        label_col = None
        run_regression(path, label_col, max_depth=3)

if __name__ == "__main__":
    main()
