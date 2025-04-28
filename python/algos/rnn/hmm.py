import logging

import torch
import torch.nn.functional as F
from torch import nn

from algos.rnn.config import Config
from algos.rnn.embedding import GaussianEmbedding
from algos.rnn.utils import get_device, logmatexp

logger = logging.getLogger(__name__)


class HMM(torch.nn.Module):
    def __init__(self, batch_size, sequence_length, num_states):
        super(HMM, self).__init__()

        self.device = get_device()
        self.num_states = num_states
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        # NB torch.randn samples the standard normal (per state).
        self.log_initial_probs = torch.nn.Parameter(
            torch.randn(num_states, dtype=torch.float32, device=self.device)
        )

        # TODO replace with transfer layer.
        self.transfer = torch.nn.Parameter(
            torch.exp(
                torch.randn(
                    num_states, num_states, dtype=torch.float32, device=self.device
                )
            )
        )

        self.embedding = GaussianEmbedding()

        # NB scratch space.
        self.ln_fs = torch.zeros(
            self.batch_size,
            self.sequence_length,
            self.num_states,
            device=self.device,
        )

        self.ln_bs = torch.zeros(
            self.batch_size,
            self.sequence_length,
            self.num_states,
            device=self.device,
        )
        
    def forward(self, obvs):
        # NB [batch_size, sequence_length, num_states]
        ln_emission_probs = self.embedding.forward(obvs.unsqueeze(-1))

        self.ln_fs[:, 0, :] = ln_emission_probs[:, 0, :] + self.log_initial_probs

        for ii in range(1, self.sequence_length):
            self.ln_fs[:, ii, :] = ln_emission_probs[:, ii, :] + logmatexp(
                self.transfer, self.ln_fs[:, ii - 1, :]
            )

        self.ln_bs[:, -1, :] = self.log_initial_probs

        for ii in range(self.sequence_length - 2, -1, -1):
            self.ln_bs[:, ii, :] = logmatexp(
                self.transfer,
                self.ln_bs[:, ii + 1, :] + ln_emission_probs[:, ii + 1, :],
            )

        log_gamma = self.ln_fs + self.ln_bs
        log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=2, keepdim=True)

        return log_gamma
