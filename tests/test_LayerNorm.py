import pytest
import numpy as np
from mini_torch import xp, asnumpy
from LayerNorm import LayerNorm

def test_layer_norm_forward_backward():
    """Tests LayerNorm's forward mean/var normalization and backwards shapes."""
    xp.random.seed(42)
    B, T, D = 2, 4, 8
    x = xp.random.randn(B, T, D)
    
    ln = LayerNorm(D)
    out = ln.forward(x)
    
    assert out.shape == (B, T, D)
    
    # Verify that the output is normalized: mean ~ 0, variance ~ 1 along the feature axis
    assert np.allclose(asnumpy(xp.mean(out, axis=-1)), 0, atol=1e-5)
    # Use higher tolerance here because variance normalization is done using variance + eps,
    # where eps is by default 1e-5.
    assert np.allclose(asnumpy(xp.var(out, axis=-1)), 1, atol=1e-4)
    
    # Backward pass checks
    grad_out = xp.random.randn(B, T, D)
    dx = ln.backward(grad_out)
    
    assert dx.shape == (B, T, D)
    assert ln.dgamma.shape == (D,)
    assert ln.dbeta.shape == (D,)
    
    assert len(ln.parameters()) == 2
    assert len(ln.grads()) == 2