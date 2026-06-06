import pytest
import numpy as np
from mini_torch import xp, asnumpy
from CausalSelfAttention import CausalSelfAttention

def test_causal_self_attention_forward_backward():
    """Tests tensor shapes and validates causal masking applied over the softmax."""
    B, T, D = 2, 4, 8
    num_heads = 2
    
    attn = CausalSelfAttention(D, num_heads)
    x = xp.random.randn(B, T, D)
    
    out = attn.forward(x)
    assert out.shape == (B, T, D)
    
    # Verify causality: The attention weights for future tokens MUST be 0.
    # attn.attn_weights shape is (B, num_heads, T, T)
    # Taking the upper triangle (excluding diagonal) should result strictly in zeros.
    mask_check = np.triu(asnumpy(attn.attn_weights)[0, 0], k=1)
    assert np.allclose(mask_check, 0.0)
    
    # Verify backward pass shapes
    grad_out = xp.random.randn(B, T, D)
    dx = attn.backward(grad_out)
    
    assert dx.shape == (B, T, D)
    assert len(attn.parameters()) == 8 # 4 Linear layers * (W and b)
    assert len(attn.grads()) == 8