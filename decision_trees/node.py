class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None, probs=None, depth=0, gain=None):
        self.feature = feature
        self.threshold = threshold
        self.gain = gain
        self.left = left
        self.right = right
        self.value = value
        self.probs = probs if probs is not None else {}
        self.depth = depth

    def __repr__(self):
        if self.value is not None:
            return f"Leaf(value={self.value}, depth={self.depth})."
        gain_str = f"{self.gain:.3f}" if isinstance(self.gain, (int, float)) else str(self.gain)
        return (f"Node(feature={self.feature}, threshold={self.threshold}, "
                f"gain={gain_str}, depth={self.depth}).")
