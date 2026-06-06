from mini_torch import Module
from mini_torch import xp

class Linear(Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        scale = xp.sqrt(2.0 / num_inputs)
        self.W = (scale * xp.random.uniform(-1.0, 1.0, size=(num_inputs, num_outputs))).astype(xp.float32)
        self.b = xp.zeros((1, num_outputs), dtype=xp.float32)
        self.x = None
        self.dW = xp.zeros_like(self.W)
        self.db = xp.zeros_like(self.b)


    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    ''' 
    # for 2D input
    def backward(self, grad_output):
        self.dW = self.x.T @ grad_output
        self.db = xp.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.W.T
    '''

    # for 3D inputs (GPT)
    def backward(self, grad_output):
      # Need to flatten x and grad_output to 2d matrices for matrix multiplication
      x_mat = self.x.reshape(-1, self.x.shape[-1])
      grad_output_mat = grad_output.reshape(-1, grad_output.shape[-1])
      
      self.dW = x_mat.T @ grad_output_mat
      self.db = xp.sum(grad_output_mat, axis=0, keepdims=True)
      return grad_output @ self.W.T

    def parameters(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]

    def zero_grad(self):
        self.dW[...] = 0.0
        self.db[...] = 0.0