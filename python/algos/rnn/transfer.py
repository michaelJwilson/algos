import torch
from torch import nn
from algos.rnn.utils import get_device, logmatexp


# @torch.compile
class DiagonalMatrixModel(nn.Module):
    def __init__(self, size, device=None):
        super(DiagonalMatrixModel, self).__init__()

        if device == None:
            device = get_device(device)
        
        self.diag = nn.Parameter(torch.randn(size, dtype=torch.float32, device=device))

    def forward(self, x):
        return logmatexp(torch.diag(self.diag), x)
