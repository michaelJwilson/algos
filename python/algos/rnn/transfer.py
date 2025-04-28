import torch
from torch import nn
from torch.optim import Adam


class DiagonalMatrixModel(nn.Module):
    def __init__(self, size):
        super(DiagonalMatrixModel, self).__init__()

        self.diagonal = nn.Parameter(torch.randn(size, dtype=torch.float32))

    def forward(self, x):
        diagonal_matrix = torch.diag(self.diagonal)

        return x @ diagonal_matrix
