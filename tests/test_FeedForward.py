import pytest
import numpy as np
from mini_torch import xp
from FeedForward import FeedForward

def test_feedforward_forward_backward():
    """Tests the sequence container effectively chains the linear and ReLU layers."""
    B, T, D = 2, 4, 8
    ffwd = FeedForward(D)
    x = xp.random.randn(B, T, D)
    
    out = ffwd.forward(x)
    assert out.shape == (B, T, D)
    
    grad_out = xp.random.randn(B, T, D)
    dx = ffwd.backward(grad_out)
    
    assert dx.shape == (B, T, D)
    assert len(ffwd.parameters()) == 4 # 2 Linear layers * (W and b)
    assert len(ffwd.grads()) == 4