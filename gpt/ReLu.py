from mini_torch import Activation

class ReLU(Activation):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0).astype(x.dtype)
        return x * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask