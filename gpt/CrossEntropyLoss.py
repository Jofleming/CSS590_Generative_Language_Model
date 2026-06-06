# Authors: Aldan Brown, Hokan Gillot
# Created 5.2.2026 for CNN assignment
# Updated 6.6.2026 for GPT assignment

from mini_torch import Loss, xp, scipy_special


class CrossEntropyLoss(Loss):
    """
    Cross-Entropy Loss module for multi-class classification.
    
    Computes the loss from logits (raw predictions) and targets for numerical stability.
    Uses scipy.special.log_softmax to avoid numerical issues with softmax outputs near zero.
    
    For one-hot encoded targets:
        loss = -sum(targets * log_softmax(logits)) / batch_size
    
    The derivative with respect to logits is:
        grad = (softmax(logits) - targets) / batch_size
    
    This assumes targets are one-hot encoded, where each row sums to 1.
    """

    def __init__(self):
        """
        Initializes the CrossEntropyLoss module.
        
        Calls the superclass constructor which sets up caches for 
        predictions and targets.
        """
        super().__init__()

    def forward(self, predictions, targets):
        """
        Computes the cross-entropy loss from logits and targets.
        
        Caches both predictions and targets for use in backward pass.
        Uses log_softmax for numerical stability to avoid log(0) errors.
        
        Args:
            predictions (numpy.ndarray): Raw network output logits of shape (batch_size, num_classes).
            targets (numpy.ndarray): One-hot encoded target labels of shape (batch_size, num_classes).
        
        Returns:
            float: The scalar cross-entropy loss.
        """
        self.predictions = predictions
        self.targets = targets
        
        # Compute log_softmax for numerical stability
        log_probs = scipy_special.log_softmax(predictions, axis=1)
        
        # Cross-entropy loss: -sum(targets * log_softmax) / batch_size
        # This works with one-hot encoded targets
        batch_loss = -xp.sum(targets * log_probs, axis=1)
        
        # Return mean loss across batch
        loss = xp.mean(batch_loss)
        
        return loss

    def backward(self):
        """
        Computes the gradient of cross-entropy loss with respect to logits.
        
        For one-hot encoded targets, the derivative is remarkably simple:
        grad = (softmax(logits) - targets) / batch_size
        
        This is derived from the chain rule between cross-entropy and softmax.
        
        Returns:
            numpy.ndarray: Gradient of loss w.r.t. logits, shape (batch_size, num_classes).
        """
        if self.predictions is None or self.targets is None:
            raise ValueError("CrossEntropyLoss.backward called before forward.")
        
        # Compute softmax of cached predictions
        probs = scipy_special.softmax(self.predictions, axis=1)
        
        # Gradient: (softmax - targets) normalized by batch size
        batch_size = self.predictions.shape[0]
        grad_output = (probs - self.targets) / batch_size
        
        return grad_output