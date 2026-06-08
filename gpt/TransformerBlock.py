from mini_torch import Module
from LayerNorm import LayerNorm
from CausalSelfAttention import CausalSelfAttention
from FeedForward import FeedForward

class TransformerBlock(Module):
      def __init__(self, emb_dim, num_heads, drop_rate=0.1):
          super().__init__()
          self.ln_1 = LayerNorm(emb_dim)
          self.attn = CausalSelfAttention(emb_dim, num_heads)
          self.ln_2 = LayerNorm(emb_dim)
          self.ffwd = FeedForward(emb_dim)

      def forward(self, x):
          
          # Attention Path with Residual
          norm1_out = self.ln_1.forward(x)
          attn_out = self.attn.forward(norm1_out)
          x1 = x + attn_out

          # FeedForward Path with Residual
          norm2_out = self.ln_2.forward(x1)
          ffwd_out = self.ffwd.forward(norm2_out)
          return x1 + ffwd_out

      def backward(self, grad_output):
          
          # Backprop through the FeedForward Residual
          grad_x1_skip = grad_output
          grad_ffwd_in = self.ffwd.backward(grad_output)
          grad_ln2_in = self.ln_2.backward(grad_ffwd_in)
          grad_x1_total = grad_x1_skip + grad_ln2_in

          # Backprop through the Attention Residual
          grad_x_skip = grad_x1_total
          grad_attn_in = self.attn.backward(grad_x1_total)
          grad_ln1_in = self.ln_1.backward(grad_attn_in)
          return grad_x_skip + grad_ln1_in

      # aggregate the parameters and gradients of its children so the Optimizer can access them
      def parameters(self):
          return (self.ln_1.parameters() + self.attn.parameters()+ self.ln_2.parameters() + self.ffwd.parameters())
      def grads(self):
          return (self.ln_1.grads() + self.attn.grads() + self.ln_2.grads() + self.ffwd.grads())