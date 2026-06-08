import numpy as np
from mini_torch import Module, xp, scipy_special, asnumpy, as_backend_array
from Embedding import Embedding
from Sequential import Sequential
from TransformerBlock import TransformerBlock
from LayerNorm import LayerNorm
from Linear import Linear         


class GPTModel(Module):
    def __init__(self, vocab_size, block_size, emb_dim, num_heads, num_layers):
          super().__init__()
          self.vocab_size = vocab_size
          self.block_size = block_size
          self.tok_emb  = Embedding(vocab_size, emb_dim)
          self.pos_emb  = Embedding(block_size, emb_dim)
          self.blocks   = Sequential([TransformerBlock(emb_dim, num_heads) for _ in range(num_layers)])
          self.ln_f     = LayerNorm(emb_dim)
          self.lm_head  = Linear(emb_dim, vocab_size)

    def forward(self, indices):
          B, T = indices.shape # Batch Size and Sequence Length

          # Step 1: Embeddings
          tok_emb    = self.tok_emb.forward(indices)                              
          pos_indices = xp.zeros((B, T), dtype=xp.int64)

          # Create a sequence of position indices and repeat it for every item in the batch so it matches the shape of indices
          for b in range(B):
            pos_indices[b] = xp.arange(T, dtype=xp.int64)

          pos_emb = self.pos_emb.forward(pos_indices)
          x = tok_emb + pos_emb

          # Step2: Transformer blocks
          x = self.blocks.forward(x)

          # Step 3: Pred head
          x = self.ln_f.forward(x)
          logits = self.lm_head.forward(x)
          return logits                                    

    def backward(self, grad_output):
          
          # Step 1: Reverse the Prediction Head
          dx = self.lm_head.backward(grad_output)
          dx = self.ln_f.backward(dx)

          # Step 2: Reverse the Transformer Blocks
          dx = self.blocks.backward(dx)

          # Step 3: Reverse the Embeddings
          self.tok_emb.backward(dx)
          self.pos_emb.backward(dx)
          return None

    def generate(self, start_indices, max_new_tokens, temperature=1.0):  
        context = start_indices                          
        for _ in range(max_new_tokens): # Loop max_new_tokens times

            # Slice context to not exceed block_size, pass it through forward to get logits, get last logits
            sliced_context = context[:, -self.block_size:] 
            logits = self.forward(sliced_context)
            last_logits = logits[:, -1, :]                                                   

            if temperature <= 0:
                probabilities = scipy_special.softmax(last_logits, axis=-1)
                next_token = xp.argmax(probabilities, axis=-1, keepdims=True) # Greedy decoding (argmax)
            else: # Sample from distribution if temp > 0
                probabilities = asnumpy(scipy_special.softmax(last_logits / temperature, axis=-1)) # asnumpy for Numpy Array so that np.random.choice works
                # On cupy, random.choice doesn't support probabilities, so we need to use numpy. See: https://docs.cupy.dev/en/stable/reference/generated/cupy.random.choice.html 
                samples = np.array([np.random.choice(self.vocab_size, p=probabilities[b]) for b in range(probabilities.shape[0])])
                next_token = as_backend_array(samples.reshape(-1, 1)) # move back to GPU
        
            context = xp.concatenate([context, next_token], axis=1) # Append newly predicted token ID to context

        return context

    def parameters(self): # concat params
          return (self.tok_emb.parameters() + self.pos_emb.parameters() + self.blocks.parameters() + self.ln_f.parameters() + self.lm_head.parameters())

    def grads(self): # concat grads
          return (self.tok_emb.grads() + self.pos_emb.grads() + self.blocks.grads() + self.ln_f.grads() + self.lm_head.grads())