import numpy as np
from mini_torch import Dataset
from CharTokenizer import CharTokenizer

class TextCorpusDataset(Dataset):
    """
    A Dataset subclass that loads a plain text file, tokenizes it at the 
    character level, and yields sliding-window (x, y) context-target pairs.
    """
    
    def __init__(self, file_path, block_size):
        """
        Initializes the dataset by loading and tokenizing the text corpus.
        
        Args:
            file_path (str): The path to the plain text file.
            block_size (int): The length of the sequence window (context length).
        """
        self.block_size = block_size
        
        # 1. Read the text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # 2. Build vocabulary and initialize tokenizer
        self.tokenizer = CharTokenizer(text)
        
        # 3. Encode the entire text into integer IDs up front
        self.data = np.array(self.tokenizer.encode(text), dtype=np.int32)
    # end method
        
    def __len__(self):
        """
        Returns the total number of valid sequences that can be extracted from the corpus.
        """
        # Total valid sequences we can extract
        return len(self.data) - self.block_size
    # end method
        
    def __getitem__(self, idx):
        """
        Retrieves a single context-target pair at the specified index.
        
        Args:
            idx (int): The index of the sequence to retrieve.
            
        Returns:
            tuple: A tuple (x, y) where x is the input sequence of shape (block_size,) 
                   and y is the target sequence of shape (block_size,), both as numpy arrays.
        """
        # x is the input sequence (context)
        x = self.data[idx : idx + self.block_size]
        # y is the exact same sequence, shifted right by 1 character
        y = self.data[idx + 1 : idx + self.block_size + 1]
        
        return x, y
    # end method
# end class