import logging

import numpy as np
import torch
from numba import njit
from torch.utils.data import Dataset

from algos.rnn.utils import get_device

"""
HMM simultation assuming Gaussian emission.
"""

logger = logging.getLogger(__name__)


@njit
def populate_states(states, transfer):
    for t in range(1, len(states)):
        ps = transfer[states[t - 1]]
        sum_ps = np.cumsum(ps)

        # NB equivalent to Categorical sampling
        #    by ps; required by njit.
        sample = np.random.rand()
        states[t] = np.searchsorted(sum_ps, sample)


@njit
def populate_obs(obs, states, means, stds, sequence_length):
    for t in range(sequence_length):
        obs[t] = means[states[t]] + stds[states[t]] * np.random.randn()


class HMMDataset(Dataset):
    def __init__(
        self,
        num_sequences,
        sequence_length,
        jump_rate,
        means,
        stds,
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
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.jump_rate = jump_rate
        self.trans = np.array(
            [[1.0 - jump_rate, jump_rate], [jump_rate, 1.0 - jump_rate]]
        )

        # NB assumes Gaussian.
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.num_states = len(self.means)

        self.states = np.zeros(self.sequence_length, dtype=int)
        self.obvs = np.zeros(self.sequence_length, dtype=float)

        logger.info(
            f"Generating HMMDataset on {self.device} with true parameters:\nM={self.means}\nT=\n{self.trans}"
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

        populate_obs(
            self.obvs, self.states, self.means, self.stds, self.sequence_length
        )

        if self.transform:
            obvs = self.transform(obvs)
            
        if self.target_transform:
            states = self.target_transform(states)
        
        states = torch.tensor(self.states, dtype=torch.long, device=self.device)
        obvs = torch.tensor(self.obvs, dtype=torch.float, device=self.device).unsqueeze(
            -1
        )

        logger.debug(f"{states}")
        logger.debug(f"Realized HMM simulation:\n{obvs}")

        # NB when called as a batch, will have shape [batch_size, seq_length, 1].
        return obvs, states
