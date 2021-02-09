import torch
from torch import nn


class NormalisedSigmoid(nn.Module):
    """ Normalised logistic sigmoid function. """

    def __init__(self, p: float = 1, dim: int = -1):
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(s)
        return torch.nn.functional.normalize(a, p=self.p, dim=self.dim)


class NormalisedReLU(nn.Module):
    """ Normalised Rectified Linear Unit. """

    def __init__(self, p: float = 1, dim: int = -1):
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        a = torch.relu(s)
        return torch.nn.functional.normalize(a, p=self.p, dim=self.dim)


class NormalisedAbs(nn.Module):
    """ Normalised absolute value function. """

    def __init__(self, p: float = 1, dim: int = -1):
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        a = torch.abs(s)
        return torch.nn.functional.normalize(a, p=self.p, dim=self.dim)
