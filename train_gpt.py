import time
import numpy as np

from mini_torch import xp, as_backend_array, asnumpy, is_gpu_available
from mini_torch import DataLoader
from TextCorpusDataset import TextCorpusDataset
from GPTModel import GPTModel
from CrossEntropyLoss import CrossEntropyLoss
from RMSprop import RMSprop

def main():
    # ---------------------------------------------------------
    # 1. Hyperparameters (Scaled down for CPU training)
    # ---------------------------------------------------------
    corpus_path = "../../Transformers/Pride-and-Prejudice-Cleaned.txt"
    block_size = 32      # Context length
    # GPUs require much larger batches to hide Python kernel launch overhead
    batch_size = 128 if is_gpu_available else 16
    emb_dim = 256 if is_gpu_available else 64         # Embedding dimension
    num_heads = 4        # Number of attention heads
    num_layers = 4 if is_gpu_available else 2       # Number of Transformer blocks
    epochs = 10 if is_gpu_available else 5
    learning_rate = 0.001
    
    # ---------------------------------------------------------
    # 2. Data Preparation
    # ---------------------------------------------------------
    print(f"Loading dataset from {corpus_path}...")
    dataset = TextCorpusDataset(corpus_path, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    vocab_size = dataset.tokenizer.vocab_size
    print(f"Dataset loaded. Vocabulary size: {vocab_size}")
    
    # ---------------------------------------------------------
    # 3. Model, Loss, and Optimizer Initialization
    # ---------------------------------------------------------
    model = GPTModel(vocab_size, block_size, emb_dim, num_heads, num_layers)
    loss_fn = CrossEntropyLoss()
    optimizer = RMSprop([model], lr=learning_rate)
    
    if is_gpu_available:
        print("Starting training loop (GPU)...")
    else:
        print("Starting training loop (CPU)...")
    # end if
    
    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    for epoch in range(epochs):
        # Start tracking time for the epoch (negligible overhead)
        epoch_start_time = time.perf_counter()
        
        total_loss = 0.0
        num_batches = len(dataloader)
        
        for batch_idx, (x, y) in enumerate(dataloader):
            # Move batch data to the active backend (GPU/CPU)
            x = as_backend_array(x)
            y = as_backend_array(y)
            
            # 1. Forward Pass
            logits = model.forward(x)
            
            # CrossEntropyLoss typically expects 2D tensors (N, Classes), 
            # so we flatten the batch and sequence dimensions before calculating loss.
            logits_flat = logits.reshape(-1, vocab_size)
            y_flat = y.reshape(-1)
            
            # One-hot encode the target indices to match the shape of logits_flat
            y_one_hot = xp.eye(vocab_size, dtype=xp.float32)[y_flat]
            
            loss = loss_fn.forward(logits_flat, y_one_hot)
            # Make sure loss is a standard float/numpy type for accumulation
            total_loss += float(asnumpy(loss))
            
            # 2. Backward Pass
            grad_logits_flat = loss_fn.backward()
            
            # Reshape gradients back to 3D (Batch, Time, Vocab) to pass into the model
            grad_logits = grad_logits_flat.reshape(x.shape[0], x.shape[1], vocab_size)
            
            optimizer.zero_grad()
            model.backward(grad_logits)
            
            # 3. Update Weights
            optimizer.step()
        # end for
            
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_loss = total_loss / num_batches
        
        print(f"Epoch {epoch + 1}/{epochs} | Avg Loss: {avg_loss:.4f} | Time: {epoch_duration:.2f}s")
        
        # ---------------------------------------------------------
        # 5. Generation (Autoregressive sample output)
        # ---------------------------------------------------------
        # Seed the generation with a simple newline character to let the model "breathe"
        start_idx = dataset.tokenizer.encode('\n')
        start_context = xp.array([start_idx]) # Shape: (1, 1)
        
        generated_indices = model.generate(start_context, max_new_tokens=40)
        generated_text = dataset.tokenizer.decode(asnumpy(generated_indices)[0].tolist())
        print(f"--- Generation Sample ---\n{generated_text}\n-------------------------\n")
     # end for
        
    model.save_weights("gpt_weights.pkl")
    print("Training complete. Weights saved to gpt_weights.pkl.")
# end main()

if __name__ == "__main__":
    main()