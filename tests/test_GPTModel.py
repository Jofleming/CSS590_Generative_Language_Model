import pytest
import numpy as np
from mini_torch import xp
from GPTModel import GPTModel

def test_gpt_model_forward_backward():
    """Verifies the integration of all components into the full GPT architecture."""
    vocab_size = 10
    block_size = 5
    emb_dim = 8
    num_heads = 2
    num_layers = 2
    B, T = 2, 4  # Note: T < block_size is perfectly valid
    
    model = GPTModel(vocab_size, block_size, emb_dim, num_heads, num_layers)
    indices = xp.random.randint(0, vocab_size, (B, T))
    
    logits = model.forward(indices)
    assert logits.shape == (B, T, vocab_size)
    
    grad_out = xp.random.randn(B, T, vocab_size)
    dx = model.backward(grad_out)
    
    # The chain rule stops here because indices are discrete
    assert dx is None
    assert len(model.parameters()) == len(model.grads())

def test_gpt_model_generate():
    """Tests the autoregressive generation loop successfully appends new tokens."""
    model = GPTModel(vocab_size=10, block_size=5, emb_dim=8, num_heads=2, num_layers=1)
    start_context = xp.random.randint(0, 10, (1, 2))  # Start with 2 tokens
    
    generated = model.generate(start_context, max_new_tokens=3)
    assert generated.shape == (1, 5) # 2 original + 3 newly generated