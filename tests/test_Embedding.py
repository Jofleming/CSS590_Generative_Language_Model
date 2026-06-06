import pytest
import numpy as np
from mini_torch import xp, asnumpy
from Embedding import Embedding

def test_embedding_forward_backward():
    """Tests forward shape and confirms backward properly sums duplicate indices."""
    vocab_size = 10
    emb_dim = 8
    B, T = 2, 3
    
    emb = Embedding(vocab_size, emb_dim)
    # Create indices where the token '1' and '2' appear multiple times
    indices = xp.array([[1, 2, 1], 
                        [0, 9, 2]])
    
    out = emb.forward(indices)
    assert out.shape == (B, T, emb_dim)
    
    # Feed a dummy gradient of all ones to make accumulation easy to trace
    grad_out = xp.ones((B, T, emb_dim), dtype=xp.float32)
    dx = emb.backward(grad_out)
    
    # The chain rule stops here
    assert dx is None
    
    # Because '1' appeared twice, its gradient should have accumulated (summed to 2)
    assert np.array_equal(asnumpy(emb.dweight[1]), np.full(emb_dim, 2.0))
    
    # '2' appeared twice as well
    assert np.array_equal(asnumpy(emb.dweight[2]), np.full(emb_dim, 2.0))
    
    # '3' never appeared, gradient should remain exactly 0
    assert np.array_equal(asnumpy(emb.dweight[3]), np.zeros(emb_dim))