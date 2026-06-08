from mini_torch import Module, xp


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        scale = 0.02
        self.weight = (scale * xp.random.randn(num_embeddings, embedding_dim)).astype(xp.float32)
        self.dweight = xp.zeros_like(self.weight)
        self.indices = None

    def forward(self, x):
        self.indices = x.astype(xp.int64)
        return self.weight[self.indices]

    def backward(self, grad_output):
        self.dweight[...] = 0.0
        xp.add.at(self.dweight, self.indices, grad_output)
        return None

    def parameters(self):
        return [self.weight]

    def grads(self):
        return [self.dweight]
