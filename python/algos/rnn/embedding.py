import torch
import logging
import torch.nn.functional as F
from torch import nn
from algos.rnn.utils import get_device
from algos.rnn.config import load_config

logger = logging.getLogger(__name__)


class GaussianEmbedding(nn.Module):
    def __init__(self, num_states, device=None):
        """
        Args:
            num_states (int): Number of Gaussian-distributed states.
        """
        super(GaussianEmbedding, self).__init__()

        self.device = get_device() if device is None else device

        self.num_states = num_states

        # NB Trainable parameters: mean and log(variance) for each state
        # self.means = torch.randn(num_states, dtype=torch.float32).to(self.device)

        # NB fixed means.
        self.means = torch.tensor([3.0, 8.0], dtype=torch.float32).to(self.device)

        # NB fixed, unit variances.
        self.log_vars = torch.zeros(num_states, dtype=torch.float32).to(self.device)

        self.means = nn.Parameter(self.means, requires_grad=True)
        self.log_vars = nn.Parameter(self.log_vars, requires_grad=False)

        logger.info(
            f"Initialized Gaussian embedding with mean={self.means} (grad? {self.means.requires_grad}) and log vars={self.log_vars} (grad? {self.log_vars.requires_grad})"
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, num_features), representing the sequence of features.

        Returns:
            torch.Tensor: Negative log-probabilities for each state and each value in the sequence,
                          shape (batch_size, sequence_length, num_states).
        """
        batch_size, sequence_length, _ = x.shape

        variances = torch.exp(self.log_vars)

        # NB expand means and variances to match the sequence: (batch_size, 1, num_states); i.e. batch_size and sequence of 1.
        means_broadcast = self.means.view(1, 1, self.num_states)
        variances_broadcast = variances.view(1, 1, self.num_states)

        # NB expand x to match the number of states: (batch_size, sequence_length, num_states)
        #    a view of original memory; -1 signifies no change;
        x_broadcast = x.expand(-1, -1, self.num_states)

        # NB log of normalization constant
        norm = 0.5 * torch.log(2 * torch.pi * variances_broadcast)

        # TODO BUG device issue: self.means on cpu, not mps??
        # NB compute negative log-probabilities for each state and each value in the sequence
        neg_log_probs = (
            norm + ((x_broadcast - means_broadcast) ** 2) / variances_broadcast
        )

        # TODO normalize with ~logsumexp.
        # neg_log_probs = torch.log(F.softmin(neg_log_probs, dim=-1))

        # NB shape = (batch_size, sequence_length, num_states)
        return neg_log_probs
