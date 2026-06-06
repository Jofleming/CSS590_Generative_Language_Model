import pytest
import numpy as np
from TextCorpusDataset import TextCorpusDataset

def test_text_corpus_dataset_sliding_window(tmp_path):
    """Tests the sliding window logic and tensor shapes returned by the dataset."""
    # 1. Create a dummy text file using pytest's tmp_path fixture
    dummy_text = "hello world, this is a test corpus."
    file_path = tmp_path / "dummy.txt"
    file_path.write_text(dummy_text, encoding="utf-8")
    
    # 2. Instantiate Dataset
    block_size = 5
    dataset = TextCorpusDataset(str(file_path), block_size)
    
    # Total length should be total_chars - block_size
    assert len(dataset) == len(dummy_text) - block_size
    
    # 3. Test getitem
    x, y = dataset[0]
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert x.shape == (block_size,)
    assert y.shape == (block_size,)
    
    # 4. Verify autoregressive shift logic
    # x[1:] and y[:-1] should be exactly identical because y is x shifted right by 1
    assert np.array_equal(x[1:], y[:-1])