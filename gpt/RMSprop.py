from mini_torch import Optimizer
from mini_torch import xp

class RMSprop(Optimizer):
    # gamma: decay rate of moving average of squared gradients
    # eps: small constant to prevent division by 0 
    def __init__(self, modules, lr=2e-4, gamma=0.99, eps=1e-8):
        super().__init__(modules, lr=lr)
        self.gamma = gamma
        self.eps = eps
        self.v = None

    def step(self):
        if self.v is None:
            self.v = [
                [xp.zeros_like(p) for p in module.parameters()]
                for module in self.modules
            ]

        for m_idx, module in enumerate(self.modules):
            params = module.parameters()
            grads  = module.grads()
            if not params:
                continue

            while len(self.v[m_idx]) < len(params):
                self.v[m_idx].append(xp.zeros_like(params[len(self.v[m_idx])]))

            for p_idx, (p, g) in enumerate(zip(params, grads)):
                if g is None:
                    continue
                # 1. moving average of squared gradient
                self.v[m_idx][p_idx] *= self.gamma
                self.v[m_idx][p_idx] += (1 - self.gamma) * (g * g)

                # 2. parameter update
                p -= self.lr * g / (xp.sqrt(self.v[m_idx][p_idx]) + self.eps)
