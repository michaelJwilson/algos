import math
import logging

import numpy as np
import torch
from abc import ABC, abstractmethod
from numba import njit
from algos.rnn.config import Config
from torch.utils.data import Dataset
from scipy.stats import nbinom, beta
from algos.rnn.utils import get_device, set_precision

"""
HMM simultation assuming Gaussian emission.
"""

logger = logging.getLogger(__name__)

set_precision()


class Emission(ABC):
    """
    Abstract base class for HMM emission models.
    """
    @abstractmethod
    def populate_obvs(*args):
        pass


@njit
def populate_states(states, transfer):
    for t in range(1, len(states)):
        ps = transfer[states[t - 1]]
        sum_ps = np.cumsum(ps)

        # NB equivalent to Categorical sampling
        #    by ps; required by njit.
        sample = np.random.rand()
        states[t] = np.searchsorted(sum_ps, sample)


class NormalEmission(Emission):
    def __init__(self):
        config = Config()

        self.means = np.array(config.gaussian_means)
        self.stds = np.array(config.gaussian_stds)

    @staticmethod
    @njit
    def populate_obvs(obvs, states, means, stds, sequence_length):
        for t in range(sequence_length):
            obvs[t] = means[states[t]] + stds[states[t]] * np.random.randn()


class NBinomialEmission(Emission):
    def __init__(self):
        config = Config()

        self.eff_coverages = np.array(config.nb_eff_coverages)
        self.sampling = np.array(config.nb_sampling)

    @staticmethod
    def populate_obvs(obvs, states, eff_coverages, samplings, sequence_length):
        for t in range(sequence_length):
            state = states[t]
            obvs[t] = nbinom.rvs(eff_coverages[state], samplings[state])


class BetaBinomEmission(Emission):
    def __init__(self):
        config = Config()

        self.alphas = np.array(config.bb_alphas)
        self.betas = np.array(config.bb_betas)

    @staticmethod
    def populate_obvs(obvs, states, alphas, betas, sequence_length):
        for t in range(sequence_length):
            state = states[t]
            pp = beta.rvs(alphas[state], betas[state])

            obvs[t] = np.random.binomial(1, pp)


class HMMDataset(Dataset):
    def __init__(
        self,
        num_states,
        num_sequences,
        sequence_length,
        jump_rate,
        emission="normal",
        device=None,
        transform=None,
        target_transform=None,
    ):
        """
        Args:
            num_sequences (int): Number of sequences in the dataset.
            sequence_length (int): Length of each sequence.
            trans (np.ndarray): Transition matrix of shape (num_states, num_states).
            means (list): List of means for Gaussian emissions for each state.
            stds (list): List of standard deviations for Gaussian emissions for each state.
        """
        self.device = get_device(device)
        self.num_states = num_states
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.emission = emission
        self.jump_rate = jump_rate
        self.trans = np.array(
            [[1.0 - jump_rate, jump_rate], [jump_rate, 1.0 - jump_rate]]
        )
        
        self.states = np.zeros(self.sequence_length, dtype=int)
        self.obvs = np.zeros(self.sequence_length, dtype=np.float32)

        self.transform = transform
        self.target_transform = target_transform

        logger.info(
            f"Generating HMMDataset on {self.device} with {emission} emission."
        )

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Generate a single HMM sequence (on the fly).
        """
        # TODO necessary?
        self.states[:] = 0
        self.obvs[:] = 0.0

        # NB uniform categorical prior on starting state.
        self.states[0] = np.random.choice(self.num_states)

        populate_states(self.states, self.trans)

        match self.emission:
            case "normal":
                em = NormalEmission()
                em.populate_obvs(self.obvs, self.states, em.means, em.stds, len(self.states))
            case "nbinom":
                em = NBinomialEmission()
                em.populate_obvs(self.obvs, self.states, em.eff_coverages, em.samplings, len(self.states))
            case "betabinom":
                em = BetaBinomEmission()
                em.populate_obvs(self.obvs, self.states, em.alphas, em.betas, len(self.states))
        """
        populate_obs_normal(
            self.obvs, self.states, self.means, self.stds, self.sequence_length
        )
        """
        if self.transform:
            self.obvs = self.transform(self.obvs)

        if self.target_transform:
            self.states = self.target_transform(self.states)

        states = torch.tensor(
            self.states, dtype=torch.int, device="cpu", pin_memory=False
        )

        obvs = torch.tensor(self.obvs, device="cpu", pin_memory=False).unsqueeze(-1)

        logger.debug(f"{states}")
        logger.debug(f"Realized HMM simulation:\n{obvs}")

        # NB when called as a batch, will have shape [batch_size, seq_length, 1].
        return obvs, states
