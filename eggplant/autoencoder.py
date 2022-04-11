import torch as t
from torch.utils.data import Dataset


class AutoEncoder(t.nn.Module):
    def __init__(self, n_features, n_hidden_1: int = 64, n_latent: int = 10):
        super().__init__()
        self.n_features = n_features
        self.encoder = t.nn.Sequential(
            t.nn.Linear(self.n_features, n_hidden_1),
            t.nn.ReLU(),
            t.nn.Linear(n_hidden_1, n_latent),
        )

        self.decoder = t.nn.Sequential(
            t.nn.Linear(n_latent, n_hidden_1),
            t.nn.ReLU(),
            t.nn.Linear(n_hidden_1, self.n_features),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoderDataSet(Dataset):
    def __init__(self, X):
        self.X = t.tensor(X)
        self.N = X.shape[0]
        self.F = X.shape[1]

    def __getitem__(self, idx) -> t.tensor:
        return self.X[idx, :]

    def __len__(
        self,
    ) -> int:
        return self.N
