import os
import numpy as np

from mini_torch import xp, as_backend_array, asnumpy, is_gpu_available
from CharTokenizer import CharTokenizer
from GPTModel import GPTModel

def main():
    corpus_path = "../../Transformers/Pride-and-Prejudice-Cleaned.txt"
    weights_path = "gpt_weights.pkl.npz"
    
    # Ensure the required files exist
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found at {corpus_path}. Required to rebuild tokenizer.")
        return
        
    if not os.path.exists(weights_path):
        print(f"Error: Weights file not found at {weights_path}. Run train_gpt.py first.")
        return

    # 1. Rebuild the tokenizer to ensure matching vocabulary
    print(f"Rebuilding tokenizer from {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size

    # 2. Hyperparameters (Must exactly match the trained model)
    block_size = 32      
    emb_dim = 256 if is_gpu_available else 64         
    num_heads = 4        
    num_layers = 4 if is_gpu_available else 2       
    
    # 3. Instantiate model and load weights
    print("Initializing model and loading weights...")
    model = GPTModel(vocab_size, block_size, emb_dim, num_heads, num_layers)
    model.load_weights(weights_path)
        
    # 4. Generate Text
    max_tokens = 400
    print(f"Generating {max_tokens} characters...\n")
    # Seed the generation with a newline character
    start_idx = tokenizer.encode('\n')
    start_context = xp.array([start_idx]) # Shape: (1, 1)
    
    # Run the autoregressive loop
    generated_indices = model.generate(start_context, max_new_tokens=max_tokens, temperature=0.8)
    
    # Decode the resulting integer array back into a string
    generated_text = tokenizer.decode(asnumpy(generated_indices)[0].tolist())
    print(generated_text)

if __name__ == "__main__":
    main()