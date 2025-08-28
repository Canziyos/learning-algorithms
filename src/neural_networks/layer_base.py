class Layer:
    def forward(self, inputs):
        """Compute outputs for this layer given inputs."""
        raise NotImplementedError
    
    def backward(self, grads):
        """Backpropagate gradients through this layer (if applicable)."""
        # Default: passthrough for layers with no params.
        return grads
    
    def has_params(self):
        """Return True if layer has trainable parameters."""
        return False

    def describe(self):
        """Return a short description of this layer."""
        return self.__class__.__name__
