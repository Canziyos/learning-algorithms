# flatten.py
from layer_base import Layer
from utils import dprint

def _get_shape_py(x):
    import numpy as np
    if isinstance(x, np.ndarray):
        return tuple(x.shape)
    if not isinstance(x, (list, tuple)):
        return ()
    if len(x) == 0:
        return (0,)
    return (len(x),) + _get_shape_py(x[0])

def _flatten_py(x):
    import numpy as np
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if not isinstance(x, (list, tuple)):
        return [x]
    flat = []
    for el in x:
        flat.extend(_flatten_py(el))
    return flat

def _reshape_py(flat, shape):
    if shape == ():
        return flat.pop(0)
    size = shape[0]
    return [_reshape_py(flat, shape[1:]) for _ in range(size)]

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.original_shape = None
        self._was_numpy = False

    def forward(self, inputs):
        import numpy as np
        self._was_numpy = isinstance(inputs, np.ndarray)
        self.original_shape = _get_shape_py(inputs)
        flat = _flatten_py(inputs)
        dprint(2, f"[Flatten] in_shape={self.original_shape} -> out_len={len(flat)}")
        return flat

    def backward(self, grads):
        flat = list(grads)  # copy
        reshaped = _reshape_py(flat, self.original_shape)
        if self._was_numpy:
            import numpy as np
            reshaped = np.array(reshaped)
        return reshaped

    def has_params(self): return False
    def describe(self): return "Flatten"