from layer_base import Layer

# === Helpers ===
def get_shape(x):
    """Return the nested list shape as a tuple."""
    if not isinstance(x, list):
        return ()
    if len(x) == 0:
        return (0,)
    return (len(x),) + get_shape(x[0])

def flatten_recursive(x):
    """Flatten nested lists into a 1D list."""
    if not isinstance(x, list):
        return [x]
    flat = []
    for elem in x:
        flat.extend(flatten_recursive(elem))
    return flat

def reshape_recursive(flat, shape):
    """Reshape flat list back into nested list with given shape."""
    if shape == ():  # scalar
        return flat.pop(0)
    size = shape[0]
    return [reshape_recursive(flat, shape[1:]) for _ in range(size)]

# === Flatten Layer ===
class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.original_shape = None

    def forward(self, inputs):
        self.original_shape = get_shape(inputs)
        return flatten_recursive(inputs)

    def backward(self, grads):
    
        # copy to avoid mutating caller’s list
        flat_copy = grads[:]
        return reshape_recursive(flat_copy, self.original_shape)

    def has_params(self):
        return False

    def describe(self):
        return "Flatten"
