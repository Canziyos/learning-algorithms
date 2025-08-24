import pandas as pd
from .tree import build_tree, predict_one, predict_many, accuracy

# Load dataset
data_pd = pd.read_csv("data/icecream.csv")
weather = data_pd["weather"].tolist()
temp = data_pd["temp"].tolist()
labels = data_pd["buy_icecream"].tolist()

features = {"weather": weather, "temp": temp}

# Build tree
tree = build_tree(features, labels, max_depth=3)
print(tree)

# Predict one
print("Prediction:", predict_one(tree, {"weather": 0, "temp": 22}))

# Predict many
samples = [{"weather": w, "temp": t} for w, t in zip(weather, temp)]
y_pred = predict_many(tree, samples)
print("Accuracy:", accuracy(labels, y_pred))
