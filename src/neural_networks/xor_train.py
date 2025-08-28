from py.model import Model
from py.dense import Dense
from dropout import DropoutLayer
from utils import MSE
from optim import update
from activations import sigmoid, sigmoid_prime, tanh, tanh_prime

# ----------------------------
# XOR Dataset
# ----------------------------
dataset = [
   ([0,0], [0]),
   ([0,1], [1]),
   ([1,0], [1]),
   ([1,1], [0])
]

num_epochs = 2000 
lr = 0.5

# ----------------------------
# Model (with Dropout)
# ----------------------------
net = Model([
    Dense(input_s=2, n_neurons=4, activation=tanh, activation_prime=tanh_prime, init="xavier"), 
    DropoutLayer(drop_prob=0.2),
    Dense(input_s=4, n_neurons=1, activation=sigmoid, activation_prime=sigmoid_prime, init="xavier")
])

net.summary()

# ----------------------------
# Training
# ----------------------------
print("\n--- Training started ---\n")

for epoch in range(num_epochs):
    total_loss = 0
    for x, y_true in dataset:
        y_pred = net.forward(x)
        loss = MSE(y_true, y_pred)

        grads = net.backward(y_true, y_pred)
        update(net, grads, lr)   # should now SKIP Dropout cleanly

        total_loss += loss

    if epoch % 200 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch}, Avg Loss: {total_loss / len(dataset)}")

print("\n--- Training finished ---\n")

# ----------------------------
# Evaluation (disable dropout)
# ----------------------------
for layer in net.layers:
    if hasattr(layer, "training"):
        layer.training = False

print("\nFinal XOR predictions:")
for x, y_true in dataset:
    y_pred = net.forward(x)
    print(f"{x} -> Predicted: {y_pred}, Expected: {y_true}")
