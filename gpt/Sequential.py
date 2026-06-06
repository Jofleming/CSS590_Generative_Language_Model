# Authors: Aldan Brown, Hokan Gillot
# Created 4.18.2026 for CNN assignment
# Updated 6.6.2026 for GPT assignment

from mini_torch import Module

class Sequential(Module):
   """
   A simple Sequential container that stacks layers and automatically
   handles forward and backward propagation.
   """

   def __init__(self, layers):
      """
      Initializes the Sequential container.
      """
      super().__init__()
      self.layers = layers

   def forward(self, x):
      """
      Performs forward pass through all layers in sequence.
      """
      for layer in self.layers:
         x = layer.forward(x)
      return x

   def backward(self, grad_output):
      """
      Performs backward pass through all layers in reverse order.
      """
      for layer in reversed(self.layers):
         grad_output = layer.backward(grad_output)
      return grad_output

   def parameters(self):
      """
      Collects parameters from all layers.
      """
      params = []
      for layer in self.layers:
         if hasattr(layer, "parameters"):
            params.extend(layer.parameters())
      return params

   def grads(self):
      """
      Collects gradients from all layers.
      """
      grads = []
      for layer in self.layers:
         if hasattr(layer, "grads"):
            grads.extend(layer.grads())
      return grads

   def zero_grad(self):
      """
      Clears gradients from all layers.
      """
      for layer in self.layers:
         if hasattr(layer, "zero_grad"):
            layer.zero_grad()