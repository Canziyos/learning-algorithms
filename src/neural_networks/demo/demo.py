import random
from conv1d import Conv1D
from dense import Dense
from model import Model
import optim
from flatten import Flatten
from pool import Pool
import utils

utils.DEBUG_LEVEL = 0   # 0 = silent, 1 = optimizer summaries, 2 = full chaos

# === 1. Generate dataset ===
def make_dataset(n_samples=200, length=6):
    data = []
    for _ in range(n_samples):
        seq = [random.randint(-5, 5) for _ in range(length)]

        # Label = 1 if there exists a rise of size >= 2 somewhere
        label = 0
        for i in range(len(seq) - 1):
            if seq[i+1] - seq[i] >= 2:
                label = 1
                break
        y = [label]  # single output neuron expects list
        data.append((seq, y))
    return data

dataset = make_dataset(200)
train, test = dataset[:150], dataset[150:]

# === 2. Define model ===
conv = Conv1D(n_filters=2, filter_s=2, step_s=2, padding=1, activation="relu")
pool = Pool(pool_s=2, step_s=2, mode="avg", dim=1)

# Compute Dense input size dynamically (Conv → Pool → Flatten)
out_len_conv = utils.win_num(len(train[0][0]), conv.padding, conv.filter_s, conv.step_s)
out_len_pool = ((out_len_conv - pool.pool_s) // pool.step_s) + 1
dense_input_size = conv.n_filters * out_len_pool

dense = Dense(input_s=dense_input_size, n_neurons=1, activation="sigmoid")

net = Model([conv, pool, Flatten(), dense])

# === 3. Training loop ===
epochs = 20
lr = 0.1

for epoch in range(epochs):
    grads, loss = utils.batch_backward(net, train, utils.MSE)
    optim.update(net, grads, lr=lr, momentum=0.9, clip=None, weight_decay=0.0)

    # Evaluate on test set
    correct = 0
    for x, y_true in test:
        y_pred = net.forward(x)
        pred_label = 1 if y_pred[0] >= 0.5 else 0
        if pred_label == y_true[0]:
            correct += 1
    acc = correct / len(test)

    print(f"Epoch {epoch+1:02d}: loss={loss:.4f}, test_acc={acc:.2f}")

# === 4. Inspect learned filters ===
print("\nLearned Conv1D filters and biases:")
for f in range(conv.n_filters):
    print(f"Filter {f}: weights={conv.filters[f]}, bias={conv.biases[f]}")
