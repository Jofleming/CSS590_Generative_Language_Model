from mini_torch import Module, xp

class Dropout(Module):
    training = True  # class-level flag — toggle globally for all instances

    def __init__(self, rate):
        super().__init__()
        self.rate = rate
        self.mask = None

    def forward(self, x):
        if not Dropout.training or self.rate == 0:
            self.mask = None
            return x
        self.mask = (xp.random.random(x.shape) > self.rate).astype(x.dtype) / (1 - self.rate)
        return x * self.mask

    def backward(self, grad):
        if self.mask is None:
            return grad
        return grad * self.mask

    def parameters(self): return []
    def grads(self):      return []
