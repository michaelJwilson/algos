import logging

import torch
import torch.nn.functional as F
from torch import nn

from algos.rnn.config import Config
from algos.rnn.embedding import GaussianEmbedding
from algos.rnn.utils import get_device, logmatexp

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

        self.transition_matrix = torch.nn.Parameter(
            torch.exp(torch.randn(num_states, num_states, dtype=torch.float32, device=self.device))
        )

        self.embedding = GaussianEmbedding()

    def forward(self, obvs):
        batch_size, sequence_length, _ = obvs.unsqueeze(-1).shape

        # NB [batch_size, sequence_length, num_states]                                                                                                                                                      
        ln_emission_probs = self.embedding.forward(obvs.unsqueeze(-1))
        
        ln_fs = torch.zeros(
            batch_size,
            self.sequence_length,
            self.num_states,
            device=self.device,
        )
        
        ln_fs[:, 0, :] = ln_emission_probs[:, 0, :] + self.log_initial_probs

        for ii in range(1, self.sequence_length):
            ln_fs[:, ii, :] = ln_emission_probs[:, ii, :] + logmatexp(self.transition_matrix, ln_fs[:, ii - 1, :])
            
        ln_bs = torch.zeros(
            batch_size,
            self.sequence_length,
            self.num_states,
            device=self.device,
        )

        ln_bs[:, -1, :] = self.log_initial_probs
                    
        for ii in range(self.sequence_length - 2, -1, -1):
            ln_bs[:, ii, :] = logmatexp(
                self.transition_matrix, ln_bs[:, ii + 1, :] + ln_emission_probs[:, ii + 1, :]
            )

        log_gamma = ln_fs + ln_bs
        log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=2, keepdim=True)

        return log_gamma
