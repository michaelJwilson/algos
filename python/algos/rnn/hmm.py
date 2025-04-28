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

        self.num_states = num_states
        self.sequence_length = sequence_length

        self.log_transition_matrix = torch.nn.Parameter(
            torch.randn(num_states, num_states, dtype=torch.float32)
        )

        self.log_initial_probs = torch.nn.Parameter(
            torch.randn(num_states, dtype=torch.float32)
        )

        self.log_emission_probs = torch.nn.Parameter(
            torch.randn(num_states, dtype=torch.float32)
        )

    def forward(self, observations):
        batch_size = observations.size(0)

        alpha = torch.zeros(
            batch_size,
            self.sequence_length,
            self.num_states,
            device=observations.device,
        )

        alpha[:, 0, :] = (
            self.log_initial_probs + self.log_emission_probs[observations[:, 0]]
        )

        for t in range(1, self.sequence_length):
            alpha[:, t, :] = (
                torch.logsumexp(
                    alpha[:, t - 1, :].unsqueeze(2) + self.log_transition_matrix, dim=1
                )
                + self.log_emission_probs[observations[:, t]]
            )

        beta = torch.zeros(
            batch_size,
            self.sequence_length,
            self.num_states,
            device=observations.device,
        )

        for t in range(self.sequence_length - 2, -1, -1):
            beta[:, t, :] = torch.logsumexp(
                beta[:, t + 1, :].unsqueeze(1)
                + self.log_transition_matrix
                + self.log_emission_probs[observations[:, t + 1]].unsqueeze(1),
                dim=2,
            )

        log_gamma = alpha + beta
        log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=2, keepdim=True)

        return log_gamma
