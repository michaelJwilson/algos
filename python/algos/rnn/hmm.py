import logging

import torch
from torch import nn

from algos.rnn.embedding import GaussianEmbedding
from algos.rnn.transfer import CategoricalPrior, LeakyTransfer

logger = logging.getLogger(__name__)


# @torch.compile
class HMM(torch.nn.Module):
    def __init__(self, batch_size, sequence_length, num_states, device):
        super(HMM, self).__init__()

        self.device = device
        self.num_states = num_states
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        # TODO HACK HARDCODE jump_rate
        self.layers = nn.ModuleList(
            [
                GaussianEmbedding(),
                CategoricalPrior(self.num_states),
                LeakyTransfer(self.num_states, 0.1),
            ]
        )

    def forward_backward(self, obvs):
        # NB [batch_size, sequence_length, num_states]                                                                                                                                                     
        ln_emission_probs = -self.layers[0].forward(obvs)

        ln_fs = [interim := self.layers[1].forward(ln_emission_probs[:, 0, :])]

        for ii in range(1, self.sequence_length):
            ln_fs.append(
                interim := ln_emission_probs[:, ii, :] + self.layers[2].forward(interim)
            )

        ln_fs = torch.stack(ln_fs, dim=1)

        assert ln_fs.shape == ln_emission_probs.shape

        # TODO Prince suggested no emission? confirm.                                                                                                                                                      
        # NB https://pytorch.org/docs/stable/generated/torch.zeros_like.html                                                                                                                               
        ln_bs = [interim := self.layers[1].forward(torch.zeros_like(ln_emission_probs[:, 0, :]))]

        for ii in range(self.sequence_length - 2, -1, -1):
            ln_bs.append(
                interim := self.layers[2].forward(
                    interim + ln_emission_probs[:, ii + 1, :]
                )
            )

        ln_bs.reverse()

        # NB https://pytorch.org/docs/stable/generated/torch.stack.html                                                                                                                                     
        ln_bs = torch.stack(ln_bs, dim=1)

        assert ln_bs.shape == ln_emission_probs.shape

        ln_gamma = ln_fs + ln_bs

        result = ln_gamma - torch.logsumexp(ln_gamma, dim=2, keepdim=True)

        return result
        
    # TODO return likelihood by forward only.
    def forward(self, obvs):
        # NB [batch_size, sequence_length, num_states]
        ln_emission_probs = -self.layers[0].forward(obvs)

        ln_fs = [interim := self.layers[1].forward(ln_emission_probs[:, 0, :])]

        for ii in range(1, self.sequence_length):
            ln_fs.append(
                interim := ln_emission_probs[:, ii, :] + self.layers[2].forward(interim)
            )

        ln_fs = torch.stack(ln_fs, dim=1)

        assert ln_fs.shape == ln_emission_probs.shape

        return torch.sum(torch.logsumexp(ln_fs[:, -1, :], dim=-1))
