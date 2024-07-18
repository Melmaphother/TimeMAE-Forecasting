import torch.nn as nn


class StandardScaler(nn.Module):
    def __init__(self):
        super(StandardScaler, self).__init__()
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, unbiased=False, keepdim=True)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return X * self.std + self.mean

    def forward(self, X):
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler has not been fitted.")
        return self.transform(X)
