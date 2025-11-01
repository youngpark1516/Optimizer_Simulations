import numpy as np

class GD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, x, grad):
        return x - self.lr * grad


class AdaGrad:
    def __init__(self, lr=0.1, eps=1e-7):
        self.lr = lr
        self.eps = eps
        self.G = 0.0

    def step(self, x, grad):
        self.G += grad**2
        adj_lr = self.lr / (np.sqrt(self.G) + self.eps)
        return x - adj_lr * grad


class RMSProp:
    def __init__(self, lr=0.01, rho=0.9, eps=1e-7):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.Eg = 0.0

    def step(self, x, grad):
        self.Eg = self.rho * self.Eg + (1 - self.rho) * grad**2
        adj_lr = self.lr / (np.sqrt(self.Eg) + self.eps)
        return x - adj_lr * grad


class Adam:
    def __init__(self, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = 0.0
        self.v = 0.0
        self.t = 0

    def step(self, x, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
