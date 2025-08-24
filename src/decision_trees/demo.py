import pandas as pd
from tree import build_tree, predict_one, predict_many, accuracy

def load_dataset(path, label):
    data = pd.read_csv(path)
    labels = data[label].tolist()
    features = {col: data[col].tolist() for col in data.columns if col != label}
    return features, labels, data, label

def main():
    path = "datasets/study_exam.csv"
    label = "label"

    features, labels, df, label_col = load_dataset(path, label)

    # Build tree.
    tree = build_tree(features, labels, max_depth=3)
    print("Tree:\n", tree)

    # Predict one (first row as test sample).
    sample = {col: df[col].iloc[0] for col in df.columns if col != label_col}
    print("Prediction for first row:", predict_one(tree, sample))

    # Predict many (whole dataset).
    samples = [dict(zip([c for c in df.columns if c != label_col], row))
               for row in df.drop(columns=[label_col]).values]
    y_pred = predict_many(tree, samples)
    print("Accuracy on dataset:", accuracy(labels, y_pred))

if __name__ == "__main__":
    main()
