import torch
from torch import nn


@torch.compile
class DiagonalMatrixModel(nn.Module):
    def __init__(self, size):
        super(DiagonalMatrixModel, self).__init__()

        self.diag = nn.Parameter(torch.randn(size, dtype=torch.float32))

    def forward(self, x):
        return x @ torch.diag(self.diag)
