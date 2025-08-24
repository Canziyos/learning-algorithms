class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None, depth=0, gain=None):
        self.feature = feature
        self.threshold = threshold
        self.gain = gain
        self.left = left
        self.right = right
        self.value = value
        self.depth = depth

    def __repr__(self):
        if self.value is not None:
            return f"Leaf(value={self.value}, depth={self.depth})"
        return (f"Node(feature={self.feature}, threshold={self.threshold}, "
                f"gain={self.gain:.3f}, depth={self.depth})")
