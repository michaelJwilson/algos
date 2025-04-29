import torch
import torch.nn.functional as F
from torch import nn
from algos.rnn.utils import get_device, logmatexp

@torch.compile                                                                                                                                                                                                     
class CategoricalPrior(nn.Module):
    def __init__(self, num_states, device=None):
        super(CategoricalPrior, self).__init__()

        if device == None:
            device = get_device(device)

        # NB torch.randn samples the standard normal (per state).                                                                                                                                                     
        self.num_states = num_states
        self.ln_pi = torch.nn.Parameter(
            torch.log(
                torch.ones(num_states, device=device) / num_states
            )
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  # states: {self.num_states}\n"
            f"  # parameters: {sum(p.numel() for p in self.parameters())}\n"
        )

    def forward(self, x):
        return x + torch.log(F.softmax(self.ln_pi, dim=0))


@torch.compile
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

@torch.compile
class LeakyTransfer(nn.Module):
    def __init__(self, num_states, jump_rate, device=None):
        super(LeakyTransfer, self).__init__()

        if device == None:
            device = get_device(device)

        self.num_states = num_states
        self.jump_rate = nn.Parameter(torch.tensor(jump_rate, device=device))
        self.eye = torch.eye(self.num_states, device=device)
        self.ones = torch.ones(self.num_states, self.num_states, device=device)
                
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  # states: {self.num_states}\n"
            f"  # parameters: 1\n"
        )

    def forward(self, xx):
        jump_rate_per_state = self.jump_rate / (self.num_states - 1.0)
        
        transfer = jump_rate_per_state * self.ones
        transfer -= jump_rate_per_state * self.eye
        transfer += (1.0 - self.jump_rate) * self.eye

        return logmatexp(transfer, xx)
