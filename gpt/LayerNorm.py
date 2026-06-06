from mini_torch import xp
from mini_torch import Module

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        """
        Initializes the LayerNorm module.
        
        Args:
            normalized_shape (int): The size of the feature dimension to be normalized 
                                    (e.g., the embedding dimension).
            eps (float): A small value added to the variance for numerical stability 
                         to prevent division by zero.
        """
        super().__init__()
        self.eps = eps
        
        # Learnable parameters
        self.gamma = xp.ones(normalized_shape, dtype=xp.float32)
        self.beta = xp.zeros(normalized_shape, dtype=xp.float32)
        
        # Caches
        self.x_hat = None
        self.std = None
        self.x_centered = None
        
        # Gradient placeholders
        self.dgamma = None
        self.dbeta = None
    # end method

    def forward(self, x):
        """
        Performs the forward pass of Layer Normalization.
        
        Args:
            x (numpy.ndarray): The input data batch, typically of shape (Batch_Size, Sequence_Length, Embedding_Dim).
            
        Returns:
            numpy.ndarray: The normalized output data of the exact same shape as x.
        """
        # Calculate mean and var along the embedding dimension
        mean = xp.mean(x, axis=-1, keepdims=True)
        
        self.x_centered = x - mean
        var = xp.mean(self.x_centered ** 2, axis=-1, keepdims=True)
        
        self.std = xp.sqrt(var + self.eps)
        self.x_hat = self.x_centered / self.std
        
        out = self.gamma * self.x_hat + self.beta
        return out
    # end method

    def backward(self, grad_output):
        """
        Performs the backward pass computation (manual backpropagation).
        
        Args:
            grad_output (numpy.ndarray): The gradient of the loss with respect to this layer's output, 
                                         typically of shape (Batch_Size, Sequence_Length, Embedding_Dim).
                                         
        Returns:
            numpy.ndarray: The gradient of the loss with respect to this layer's input, matching the shape of x.
        """
        # 1. Parameter Gradients 
        # Reshaping to 2D before reduction is significantly faster in NumPy 
        # than reducing over an arbitrary tuple of N-dimensional axes.
        grad_out_flat = grad_output.reshape(-1, grad_output.shape[-1])
        x_hat_flat = self.x_hat.reshape(-1, self.x_hat.shape[-1])
        
        dgamma_step = xp.sum(grad_out_flat * x_hat_flat, axis=0)
        dbeta_step = xp.sum(grad_out_flat, axis=0)
        
        if self.dgamma is None:
            self.dgamma = dgamma_step
            self.dbeta = dbeta_step
        else:
            self.dgamma += dgamma_step
            self.dbeta += dbeta_step
        # end if
        
        # 2. Input Gradient (dx)
        grad_x_hat = grad_output * self.gamma
        mean_grad_x_hat = xp.mean(grad_x_hat, axis=-1, keepdims=True)
        mean_grad_x_hat_times_x_hat = xp.mean(grad_x_hat * self.x_hat, axis=-1, keepdims=True)
        
        dx = (1.0 / self.std) * (grad_x_hat - mean_grad_x_hat - self.x_hat * mean_grad_x_hat_times_x_hat)
        
        return dx
    # end method
        
    def parameters(self):
        """
        Retrieves the layer's learnable parameters (scale and shift).
        """
        return [self.gamma, self.beta]
    # end method

    def grads(self):
        """
        Retrieves the layer's calculated gradients for its learnable parameters.
        """
        return [self.dgamma, self.dbeta]
    # end method
# end class