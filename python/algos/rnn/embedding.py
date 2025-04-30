import logging

import torch
from torch import nn

from algos.rnn.config import Config
from algos.rnn.utils import get_device

logger = logging.getLogger(__name__)


# @torch.compile
class GaussianEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        config = Config()

        self.device = get_device()
        self.num_states = config.num_states

        self.means = torch.tensor(config.init_means, device=self.device)
        self.log_vars = torch.tensor(config.init_stds, device=self.device)

        self.means = nn.Parameter(self.means, requires_grad=True)
        self.log_vars = nn.Parameter(self.log_vars, requires_grad=False)

        assert (
            len(self.means) == self.num_states
        ), "Mean initialization provided inconsistent with number of states defined."

        logger.info(
            f"Initialized Gaussian embedding on device {self.device} with mean={self.means} (grad? {self.means.requires_grad}) and log vars={self.log_vars} (grad? {self.log_vars.requires_grad})"
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  # states: {self.num_states}\n"
            f"  # parameters: {sum(p.numel() for p in self.parameters())}\n"
        )

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape
        variances = torch.exp(self.log_vars)

        # NB no-copy view; expands singleton last dimension to num_states size.
        x = x.expand(-1, -1, self.num_states)

        norm = 0.5 * torch.log(2.0 * torch.pi * variances)

        # NB shape = (batch_size, sequence_length, num_states)
        return norm + ((x - self.means) ** 2) / norm
