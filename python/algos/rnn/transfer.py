import torch
from torch import nn
from algos.rnn.utils import get_device, logmatexp


# @torch.compile
class DiagonalTransfer(nn.Module):
    def __init__(self, num_states, device=None):
        super(DiagonalTransfer, self).__init__()

        if device == None:
            device = get_device(device)

        self.num_states = num_states
        self.diag = torch.ones(num_states, device=device)
        self.diag = nn.Parameter(self.diag)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  # states: {self.num_states}\n"
            f"  # parameters: {list(self.diag.shape)}\n"
        )

    def forward(self, xx):
        return logmatexp(torch.diag(self.diag), xx)
