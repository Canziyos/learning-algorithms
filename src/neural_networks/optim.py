# optim.py
momentum_state = {}

def reset_momentum():
    """Clear all momentum slots (call this if reinit or swap models)."""
    momentum_state.clear()

def update(net, grads, lr, momentum=0.9, clip=None, weight_decay=0.0):
    """
    Apply one SGD+momentum step to all param layers.

    - lr: learning rate
    - momentum: momentum coefficient
    - clip: elementwise gradient clip threshold (None = no clipping)
    - weight_decay: decoupled L2 (layers implement it as W *= (1 - lr * wd) + v)
    """
    for l, layer in enumerate(net.layers):
        if not layer.has_params():
            continue
        if grads[l] is None or grads[l] == {}:
            # nothing to apply (non-param or skipped).
            continue

        # Each layer handles its own momentum slots keyed into 'momentum_state'.
        layer.apply_grads(
            grads[l],
            lr=lr,
            momentum=momentum,
            clip=clip,
            weight_decay=weight_decay,
            state=momentum_state,
            layer_key=l,
        )
