import logging

import torch
import numpy as np
from torch import nn

from algos.rnn.config import Config
from algos.rnn.utils import get_device, torch_compile

logger = logging.getLogger(__name__)


@torch_compile
class GaussianEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        config = Config()

        self.device = get_device()
        self.num_states = config.num_states

        self.means = torch.tensor(config.init_gaussian_means, device=self.device)
        self.log_vars = torch.tensor(
            2.0 * np.log(config.init_gaussian_stds).astype(np.float32), device=self.device
        )

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


@torch_compile
class NegativeBinomialEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        config = Config()

        self.device = get_device()
        self.num_states = config.num_states

        self.eff_coverage = torch.tensor(
            config.init_nb_eff_coverage, device=self.device
        )
        self.logits = torch.tensor(config.init_nb_logits, device=self.device)

        self.eff_coverage = nn.Parameter(self.eff_coverage, requires_grad=True)
        self.logits = nn.Parameter(self.logits, requires_grad=True)

        assert (
            len(self.logits) == self.num_states
        ), "Logits initialization provided inconsistent with number of states defined."

        logger.info(
            f"Initialized Negative Binomial embedding on device {self.device} with logits={self.logits} (grad? {self.logits.requires_grad}) and coverage={self.eff_coverage} (grad? {self.eff_coverage.requires_grad})"
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  # states: {self.num_states}\n"
            f"  # parameters: {sum(p.numel() for p in self.parameters())}\n"
        )

    def forward(self, k):
        batch_size, sequence_length, _ = k.shape

        k = k.expand(-1, -1, self.num_states)
        p = torch.sigmoid(self.logits)

        log_prob = (
            torch.lgamma(k + self.eff_coverage)
            - torch.lgamma(self.eff_coverage)
            - torch.lgamma(k + 1)
            + self.eff_coverage * torch.log(1.0 - p)
            + k * torch.log(p)
        )

        # NB shape = (batch_size, sequence_length, num_states)
        return log_prob


@torch_compile
class BetaBinomialEmbedding(nn.Module):
    def __init__(self, coverage):
        super().__init__()

        config = Config()

        self.device = get_device()
        self.num_states = config.num_states

        self.alphas = torch.tensor(config.init_bb_alphas, device=self.device)
        self.betas = torch.tensor(config.init_bb_betas, device=self.device)

        # NB no-op if a tensor and on the correct device.  Otherwise, warn?
        self.coverage = torch.tensor(coverage, device=self.device)

        self.alphas = nn.Parameter(self.alphas, requires_grad=True)
        self.betas = nn.Parameter(self.betas, requires_grad=True)

        assert (
            len(self.alphas) == self.num_states
        ), "Alpha initialization provided inconsistent with number of states defined."

        logger.info(
            f"Initialized Beta Binomial embedding on device {self.device} with alpha={self.alphas} (grad? {self.alphas.requires_grad}), beta={self.betas} (grad? {self.betas.requires_grad})."
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  # states: {self.num_states}\n"
            f"  # parameters: {sum(p.numel() for p in self.parameters())}\n"
        )

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape
        x = x.expand(-1, -1, self.num_states)

        msg = f"Inconsistent coverage {self.coverage.shape} and input {x.shape}"
        
        assert self.coverage.shape[:-1] == x.shape[:-1], msg

        # NB mirrors wikipedia ordering, wrt numerator and denominator.
        log_prob = (
            torch.lgamma(self.coverage + 1)
            + torch.lgamma(x + self.alphas)
            + torch.lgamma(self.coverage - x + self.betas)
            + torch.lgamma(self.alphas + self.betas)
            - torch.lgamma(self.coverage + self.alphas + self.betas)
            - torch.lgamma(x + 1)
            - torch.lgamma(self.coverage - x + 1)
            - torch.lgamma(self.alphas)
            - torch.lgamma(self.betas)
        )

        # NB shape = (batch_size, sequence_length, num_states)
        return log_prob
