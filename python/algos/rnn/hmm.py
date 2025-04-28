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

        # NB torch.randn samples the standard normal (per state).
        self.log_initial_probs = torch.nn.Parameter(
            torch.randn(num_states, dtype=torch.float32)
        )

        self.log_transition_matrix = torch.nn.Parameter(
            torch.randn(num_states, num_states, dtype=torch.float32)
        )

        self.log_emission_probs = torch.nn.Parameter(
            torch.randn(num_states, dtype=torch.float32)
        )

    def forward(self, obvs):
        batch_size, sequence_length = obvs.shape

        one_hot_obvs = F.one_hot(obvs, num_classes=self.num_states).float()
        
        alpha = torch.zeros(
            batch_size,
            self.sequence_length,
            self.num_states,
            device=obvs.device,
        )

        beta = torch.zeros(
            batch_size,
            self.sequence_length,
            self.num_states,
            device=obvs.device,
	)

        emission_probs = one_hot_obvs @ self.log_emission_probs

        alpha[:, 0, :] = self.log_initial_probs + emission_probs[:, 0, :]
        """
        # NB initial PI prior + emission of first character in obvs.
        alpha[:, 0, :] = (
            self.log_initial_probs + self.log_emission_probs.gather(0, obvs[:, 0].unsqueeze(1)).squeeze(1)
        )
        """
        for t in range(1, self.sequence_length):
            alpha[:, t, :] = (
                torch.logsumexp(
                    alpha[:, t - 1, :].unsqueeze(2) + self.log_transition_matrix, dim=1
                )
                + self.log_emission_probs.gather(0, obvs[:, t].unsqueeze(1)).squeeze(1)
            )

        for t in range(self.sequence_length - 2, -1, -1):
            beta[:, t, :] = torch.logsumexp(
                beta[:, t + 1, :].unsqueeze(1)
                + self.log_transition_matrix
                + self.log_emission_probs[obvs[:, t + 1]].unsqueeze(1),
                dim=2,
            )

        log_gamma = alpha + beta
        log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=2, keepdim=True)

        return log_gamma
