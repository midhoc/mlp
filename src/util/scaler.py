import numpy as np

class CustomStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
    
    def transform(self, data):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted.")
        return (data - self.mean) / self.std
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def save(self, filepath):
        np.savez(filepath, mean=self.mean, std=self.std)
    
    def load(self, filepath):
        loaded = np.load(filepath)
        self.mean = loaded['mean']
        self.std = loaded['std']
        