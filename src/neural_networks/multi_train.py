import random, matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
import optim
from model import Model
from dense import Dense
from dropout import DropoutLayer
from utils import cross_entropy, batch_backward, set_training_mode
from activations import tanh, tanh_prime, softmax

# ----------------------------
# Dataset
# ----------------------------
X, y = make_circles(n_samples=300, noise=0.2, factor=0.5, random_state=42)

# standardize
X_arr = np.array(X)
mu, sigma = X_arr.mean(axis=0), X_arr.std(axis=0) + 1e-8
X = [list((x - mu) / sigma) for x in X_arr]

def one_hot(y, num_classes):
    return [[1 if i == label else 0 for i in range(num_classes)] for label in y]

Y = one_hot(y, 2)
dataset = list(zip(X, Y))

# shuffle + split 80/20
random.shuffle(dataset)
split = int(0.8 * len(dataset))
train_set, val_set = dataset[:split], dataset[split:]

# ----------------------------
# Model (Dense + Dropout)
# ----------------------------
net = Model([
    Dense(input_s=2, n_neurons=16, activation=tanh, activation_prime=tanh_prime, init="xavier"),
    DropoutLayer(drop_prob=0.3),
    Dense(input_s=16, n_neurons=8, activation=tanh, activation_prime=tanh_prime, init="xavier"),
    Dense(input_s=8, n_neurons=2, activation=softmax, activation_prime=None, init="xavier")
])

net.summary()

# ----------------------------
# Training
# ----------------------------
optim.momentum_state = {}
num_epochs = 2000
batch_size = 8
lr, momentum = 0.02, 0.8

print("\n--- Training started ---\n")
set_training_mode(net, True)

for epoch in range(num_epochs):
    random.shuffle(train_set)
    total_loss = 0.0

    for b in range(0, len(train_set), batch_size):
        batch = train_set[b:b+batch_size]
        grads, batch_loss = batch_backward(net, batch, cross_entropy)
        optim.update(net, grads, lr=lr, momentum=momentum, clip=0.8, weight_decay=1e-4)
        total_loss += batch_loss

    if epoch % 200 == 0 or epoch == num_epochs - 1:
        avg_loss = total_loss / (len(train_set) / batch_size)
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")

# ----------------------------
# Evaluation (disable dropout)
# ----------------------------
set_training_mode(net, False)

def evaluate(data):
    correct = 0
    for x, y_true in data:
        y_pred = net.forward(x)
        if y_pred.index(max(y_pred)) == y_true.index(1):
            correct += 1
    return correct / len(data)

train_acc = evaluate(train_set)
val_acc = evaluate(val_set)

print(f"\nFinal Training accuracy: {train_acc*100:.2f}%")
print(f"Validation accuracy: {val_acc*100:.2f}%")

# ----------------------------
# Decision boundary (on full dataset)
# ----------------------------
h = 0.02
X_arr = np.array([x for x, _ in dataset])
y_arr = np.array([np.argmax(y) for _, y in dataset])

x_min, x_max = X_arr[:, 0].min() - 0.5, X_arr[:, 0].max() + 0.5
y_min, y_max = X_arr[:, 1].min() - 0.5, X_arr[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = [np.argmax(net.forward(p)) for p in grid_points]
Z = np.array(Z).reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)
plt.scatter(X_arr[:, 0], X_arr[:, 1], c=y_arr, cmap=plt.cm.Spectral, edgecolors="k")
plt.title("Decision Boundary on Circles with Dropout (train vs val)")
plt.show()
