import pytest
import numpy as np
from mini_torch import xp
from TransformerBlock import TransformerBlock

def test_transformer_block_forward_backward():
    """Tests the block successfully routes data & gradients across skip connections."""
    B, T, D = 2, 4, 8
    num_heads = 2
    block = TransformerBlock(D, num_heads)
    x = xp.random.randn(B, T, D)
    
    out = block.forward(x)
    assert out.shape == (B, T, D)
    
    grad_out = xp.random.randn(B, T, D)
    dx = block.backward(grad_out)
    
    assert dx.shape == (B, T, D)
    # Parameter check: LN(2) + Attn(8) + LN(2) + FFN(4) = 16
    assert len(block.parameters()) == 16
    assert len(block.grads()) == 16