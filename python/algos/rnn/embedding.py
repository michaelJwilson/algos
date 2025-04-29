import logging

import torch
from torch import nn

from algos.rnn.config import Config
from algos.rnn.utils import get_device

logger = logging.getLogger(__name__)


class GaussianEmbedding(nn.Module):
    def __init__(self):
        super(GaussianEmbedding, self).__init__()

        self.device = get_device()
        self.num_states = Config().num_states

        # NB fixed mean initialization.
        self.means = torch.tensor([3.0, 8.0], dtype=device=self.device)

        # TODO HACK
        # NB fixed, unit variances.
        self.log_vars = 1.1 * torch.ones(self.num_states, device=self.device)

        self.means = nn.Parameter(self.means, requires_grad=True)
        self.log_vars = nn.Parameter(self.log_vars, requires_grad=False)

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

        # NB expand means and variances to match the sequence: (batch_size, 1, num_states); i.e. batch_size and sequence of 1.
        means_broadcast = self.means.view(1, 1, self.num_states)
        variances_broadcast = variances.view(1, 1, self.num_states)

        # NB expand x to match the number of states: (batch_size, sequence_length, num_states)
        #    a view of original memory; -1 signifies no change;
        x_broadcast = x.expand(-1, -1, self.num_states)
        
        # NB log of normalization constant
        norm = 0.5 * torch.log(2. * torch.pi * variances_broadcast)

        # NB compute negative log-probabilities for each state and each value in the sequence
        neg_log_probs = (
            norm + ((x_broadcast - means_broadcast) ** 2) / variances_broadcast
        )

        # NB shape = (batch_size, sequence_length, num_states)
        return neg_log_probs
