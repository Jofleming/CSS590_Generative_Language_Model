class CharTokenizer:
    """
    A simple character-level tokenizer that builds a vocabulary from a given 
    text corpus and encodes/decodes text to and from integer IDs.
    """
    def __init__(self, text):
        # Create a sorted list of all unique characters in the text
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        # Build the lookup dictionaries
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
    # end method

    def encode(self, string_data):
        # Convert a string into a list of integer IDs
        return [self.stoi[c] for c in string_data]
    # end method

    def decode(self, integer_list):
        # Convert a list of integer IDs back into a string
        return ''.join([self.itos[i] for i in integer_list])
    # end method
    
# end class