from utils import dprint

momentum_state = {}

def reset_momentum():
    momentum_state.clear()

def update(net, grads, lr, momentum=0.9, clip=1.0, weight_decay=0.0):
    global momentum_state
    for l, layer in enumerate(net.layers):
        if not layer.has_params():
            continue

        # Epoch-level summary logs (level=1).
        dprint(1, f"\n[Optimizer Update] Layer {l}: {layer.describe()}")
        dprint(1, f"Before: {layer.params()}")

        # Each layer now knows how to update itself
        layer.apply_grads(grads[l], lr, momentum, clip, weight_decay, momentum_state, l)

        dprint(1, f"After:  {layer.params()}")
