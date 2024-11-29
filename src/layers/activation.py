import numpy as np

class Tanh():
    def __init__(self):
        pass
    
    def __call__(self, x):
        return np.tanh(x)
    
    def prime(self, x):
        return 1-np.tanh(x)**2

class ReLu():
    def __init__(self):
        pass
    
    def __call__(self, x):
        return np.maximum(0, x)
    
    def prime(self, x):
        return np.where(x > 0, 1, 0)
    
class LeakyReLu():
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        
    def __call__(self, x):
        return np.where(x >= 0, x, x * self.alpha)
    
    def prime(self, x):
        return np.where(x >= 0, 1, self.alpha)
    
class sigmoid():
    def __init__(self):
        pass
        
    def __call__(self,x):
        return 1/(1+np.exp(-x))
    
    def prime(self,x):
        fx = self.__call__(x)
        return fx*(1-fx)
    

class Softmax():
    def __init__(self):
        pass
    
    def __call__(self, x):
        exp_vals = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    def prime(self, x):
        s = self.__call__(x)
        return s * (1 - s)
    
