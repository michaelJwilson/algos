import torch
import torch.nn.functional as F
from torch import nn

from algos.rnn.utils import get_device, logmatexp, torch_compile


@torch_compile
class CategoricalPrior(nn.Module):
    def __init__(self, num_states, device=None):
        super().__init__()

        device = get_device(device)

        # NB torch.randn samples the standard normal (per state).
        self.num_states = num_states

        # NB in effect, corresponds to logits normalized by softmax.
        self.ln_pi = torch.nn.Parameter(
            torch.log(torch.ones(num_states, device=device) / num_states)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  # states: {self.num_states}\n"
            f"  # parameters: {sum(p.numel() for p in self.parameters())}\n"
        )

    def forward(self, x):
        return x + torch.log(F.softmax(self.ln_pi, dim=0))


@torch_compile
class DiagonalTransfer(nn.Module):
    def __init__(self, num_states, device=None):
        super().__init__()

        device = get_device(device)

        self.num_states = num_states
        self.diag = nn.Parameter(torch.ones(num_states, device=device))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  # states: {self.num_states}\n"
            f"  # parameters: {list(self.diag.shape)}\n"
        )

    def forward(self, xx):
        return logmatexp(torch.diag(self.diag), xx)


@torch_compile
class LeakyTransfer(nn.Module):
    def __init__(self, num_states, jump_rate, device=None):
        super().__init__()

        device = get_device(device)

        self.num_states = num_states
        self.jump_rate = nn.Parameter(torch.tensor(jump_rate, device=device))
        self._eye = torch.eye(self.num_states, device=device)
        self._ones = torch.ones(self.num_states, self.num_states, device=device)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  # states: {self.num_states}\n"
            f"  # parameters: 1\n"
        )

    def forward(self, xx):
        jump_rate_per_state = self.jump_rate / (self.num_states - 1.0)

        transfer = jump_rate_per_state * self._ones
        transfer -= jump_rate_per_state * self._eye
        transfer += (1.0 - self.jump_rate) * self._eye

        return logmatexp(transfer, xx)
