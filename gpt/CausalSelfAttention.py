from mini_torch import Module, xp, scipy_special
from Linear import Linear


class CausalSelfAttention(Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError("emb_dim must be divisible by num_heads")

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.scale = xp.sqrt(xp.asarray(self.head_dim, dtype=xp.float32))

        self.q_proj = Linear(emb_dim, emb_dim)
        self.k_proj = Linear(emb_dim, emb_dim)
        self.v_proj = Linear(emb_dim, emb_dim)
        self.out_proj = Linear(emb_dim, emb_dim)

        self.x_shape = None
        self.q = None
        self.k = None
        self.v = None
        self.attn_weights = None
        self.context = None
        self.mask = None

    def _split_heads(self, x):
        B, T, D = x.shape
        return x.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, x):
        B, H, T, Hd = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * Hd)

    def forward(self, x):
        self.x_shape = x.shape
        B, T, _ = x.shape

        q_linear = self.q_proj.forward(x)
        k_linear = self.k_proj.forward(x)
        v_linear = self.v_proj.forward(x)

        self.q = self._split_heads(q_linear)
        self.k = self._split_heads(k_linear)
        self.v = self._split_heads(v_linear)

        scores = xp.matmul(self.q, self.k.transpose(0, 1, 3, 2)) / self.scale

        self.mask = xp.triu(xp.ones((T, T), dtype=bool), k=1)
        scores = xp.where(self.mask[None, None, :, :], -1.0e9, scores)

        self.attn_weights = scipy_special.softmax(scores, axis=-1).astype(x.dtype)
        self.context = xp.matmul(self.attn_weights, self.v)

        merged = self._merge_heads(self.context)
        return self.out_proj.forward(merged)

    def backward(self, grad_output):
        B, T, _ = grad_output.shape

        grad_merged = self.out_proj.backward(grad_output)
        grad_context = self._split_heads(grad_merged)

        grad_attn = xp.matmul(grad_context, self.v.transpose(0, 1, 3, 2))
        grad_v = xp.matmul(self.attn_weights.transpose(0, 1, 3, 2), grad_context)

        dot = xp.sum(grad_attn * self.attn_weights, axis=-1, keepdims=True)
        grad_scores = self.attn_weights * (grad_attn - dot)
        grad_scores = xp.where(self.mask[None, None, :, :], 0.0, grad_scores)

        grad_q = xp.matmul(grad_scores, self.k) / self.scale
        grad_k = xp.matmul(grad_scores.transpose(0, 1, 3, 2), self.q) / self.scale

        grad_q = self._merge_heads(grad_q)
        grad_k = self._merge_heads(grad_k)
        grad_v = self._merge_heads(grad_v)

        dx_q = self.q_proj.backward(grad_q)
        dx_k = self.k_proj.backward(grad_k)
        dx_v = self.v_proj.backward(grad_v)

        return dx_q + dx_k + dx_v

    def parameters(self):
        return (
            self.q_proj.parameters()
            + self.k_proj.parameters()
            + self.v_proj.parameters()
            + self.out_proj.parameters()
        )

    def grads(self):
        return (
            self.q_proj.grads()
            + self.k_proj.grads()
            + self.v_proj.grads()
            + self.out_proj.grads()
        )

    def zero_grad(self):
        self.q_proj.zero_grad()
        self.k_proj.zero_grad()
        self.v_proj.zero_grad()
        self.out_proj.zero_grad()
