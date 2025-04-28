import logging

import torch
import torch.nn.functional as F
from torch import nn

from algos.rnn.config import Config
from algos.rnn.embedding import GaussianEmbedding
from algos.rnn.utils import get_device

logger = logging.getLogger(__name__)


class HMM(torch.nn.Module):
    def __init__(self, num_states, sequence_length):
        super(HMM, self).__init__()

        self.device = get_device()
        self.num_states = num_states
        self.sequence_length = sequence_length

        # NB torch.randn samples the standard normal (per state).
        self.log_initial_probs = torch.nn.Parameter(
            torch.randn(num_states, dtype=torch.float32, device=self.device)
        )

        self.log_transition_matrix = torch.nn.Parameter(
            torch.randn(num_states, num_states, dtype=torch.float32, device=self.device)
        )

        self.embedding = GaussianEmbedding()

    def forward(self, obvs):
        batch_size, sequence_length, _ = obvs.unsqueeze(-1).shape

        alpha = torch.zeros(
            batch_size,
            self.sequence_length,
            self.num_states,
            device=self.device,
        )

        beta = torch.zeros(
            batch_size,
            self.sequence_length,
            self.num_states,
            device=self.device,
        )

        # NB [batch_size, sequence_length, num_states]
        emission_probs = self.embedding.forward(obvs.unsqueeze(-1))
        
        alpha[:, 0, :] = emission_probs[:, 0, :] + self.log_initial_probs

        for t in range(1, self.sequence_length):
            alpha[:, t, :] = torch.logsumexp(
                alpha[:, t - 1, :].unsqueeze(-1) + self.log_transition_matrix, dim=1
            ) + emission_probs[:, t, :]

        for t in range(self.sequence_length - 2, -1, -1):
            beta[:, t, :] = torch.logsumexp(
                beta[:, t + 1, :].unsqueeze(1)
                + self.log_transition_matrix
                + emission_probs[:, t+1, :],
                dim=2,
            )

        log_gamma = alpha + beta
        log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=2, keepdim=True)

        return log_gamma
