import numpy as np


class mse():
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        return np.mean(np.power(y_true-y_pred, 2))

    def prime(self, y_true, y_pred):
        return 2*(y_pred-y_true)/y_true.size


class CategoricalCrossEntropy():
    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon

    def __call__(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -np.sum(y_true * np.log(y_pred), axis=-1)

    def prime(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -y_true / y_pred


class BinaryCrossEntropy():
    def __init__(self, epsilon=1e-15):
        self.epsilon = epsilon

    def __call__(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def prime(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))
