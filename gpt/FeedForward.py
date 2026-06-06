from mini_torch import Module
from Sequential import Sequential
from Linear import Linear
from ReLU import ReLU

class FeedForward(Module):
    """
    The Feed-Forward Network module for the Transformer.
    It expands the embedding dimension by a factor of 4, applies a 
    non-linearity, and projects it back down to the original dimension.
    """
    def __init__(self, emb_dim):
        super().__init__()
        
        # The hidden dimension is 4x the embedding dimension
        hidden_dim = 4 * emb_dim
        
        # We leverage the existing Sequential container to chain the layers
        self.net = Sequential([
            Linear(emb_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, emb_dim)
        ])
    # end method

    def forward(self, x): return self.net.forward(x)

    def backward(self, grad_output): return self.net.backward(grad_output)
        
    def parameters(self): return self.net.parameters()
    def grads(self): return self.net.grads()
# end class