import pytest
from CharTokenizer import CharTokenizer

def test_char_tokenizer_vocab_size():
    """Tests that the tokenizer correctly identifies the unique vocabulary size."""
    text = "hello world"
    tokenizer = CharTokenizer(text)
    # Unique chars in 'hello world': ' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w' (8 unique)
    assert tokenizer.vocab_size == 8

def test_char_tokenizer_encode_decode():
    """Tests that encoding and then decoding returns the exact original string."""
    text = "abcdefghijklmnopqrstuvwxyz"
    tokenizer = CharTokenizer(text)
    
    original = "hello"
    encoded = tokenizer.encode(original)
    
    assert isinstance(encoded, list)
    assert len(encoded) == 5
    assert all(isinstance(x, int) for x in encoded)
    
    decoded = tokenizer.decode(encoded)
    assert decoded == original