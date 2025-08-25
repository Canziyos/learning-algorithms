from tree import ClassificationTree
from metrics import accuracy
from utils import load_dataset

def main():
    path = "datasets/study_exam.csv"
    label_col = "label"

    # Load data.
    features, labels, df, label_col = load_dataset(path, label_col)

    # Train.
    clf = ClassificationTree(max_depth=3)
    clf.fit(features, labels)

    # Inspect the tree.
    print(clf)              
    print(clf.root)

    # Prepare feature columns once.
    feature_cols = [c for c in df.columns if c != label_col]

    # Predict one (first row as test sample).
    sample = {col: df[col].iloc[0] for col in feature_cols}
    print("Prediction for first row:", clf.predict_one(sample))

    # Predict many (whole dataset).
    samples = [dict(zip(feature_cols, row)) for row in df[feature_cols].values]
    y_pred = clf.predict_many(samples)
    print("Accuracy on dataset:", accuracy(labels, y_pred))

if __name__ == "__main__":
    main()
