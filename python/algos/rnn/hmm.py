import logging

import torch
import torch.nn.functional as F
from collections import deque
from torch import nn

from algos.rnn.config import Config
from algos.rnn.embedding import GaussianEmbedding
from algos.rnn.utils import get_device, logmatexp
from algos.rnn.transfer import DiagonalMatrixModel

logger = logging.getLogger(__name__)


# @torch.compile
class CategoricalPrior(nn.Module):
    def __init__(self, num_states, device=None):
        super(CategoricalPrior, self).__init__()

        if device == None:
            device = get_device(device)

        # NB torch.randn samples the standard normal (per state).
        self.ln_pi = torch.nn.Parameter(
            torch.ones(num_states, dtype=torch.float32, device=device) / num_states
        )

    def forward(self, x):
        return x + self.ln_pi


# @torch.compile
class HMM(torch.nn.Module):
    def __init__(self, batch_size, sequence_length, num_states):
        super(HMM, self).__init__()

        self.device = get_device()
        self.num_states = num_states
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.layers = nn.ModuleList(
            [
                GaussianEmbedding(),
                CategoricalPrior(self.num_states),
                DiagonalMatrixModel(self.num_states),
            ]
        )

    def forward(self, obvs):
        # NB [batch_size, sequence_length, num_states]
        ln_emission_probs = self.layers[0].forward(obvs)

        ln_fs = [interim := self.layers[1].forward(ln_emission_probs[:, 0, :])]

        for ii in range(1, self.sequence_length):
            ln_fs.append(
                interim := ln_emission_probs[:, ii, :] + self.layers[2].forward(interim)
            )

        ln_fs = torch.stack(ln_fs, dim=1)

        # TODO Prince suggested no emission? confirm.
        ln_bs = [interim := self.layers[1].forward(ln_emission_probs[:, -1, :])]

        for ii in range(self.sequence_length - 2, -1, -1):
            ln_bs.append(
                interim := +self.layers[2].forward(
                    interim + ln_emission_probs[:, ii + 1, :]
                )
            )

        ln_bs = torch.stack(ln_bs, dim=1)

        # NB generates a copy.
        ln_bs = torch.flip(ln_bs, dims=(1,))

        log_gamma = ln_fs + ln_bs
        log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=2, keepdim=True)

        return log_gamma
