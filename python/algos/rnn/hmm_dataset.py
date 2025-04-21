import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from numba import njit

from algos.rnn.utils import get_device

logger = logging.getLogger(__name__)


@njit
def populate_states(states, num_states, transfer):
    for t in range(1, len(states)):
        ps = transfer[states[t - 1]]
        sum_ps = np.cumsum(ps)
        sample = np.random.rand()
        states[t] = np.searchsorted(sum_ps, sample)

    return


@njit
def populate_obs(obs, states, means, stds, sequence_length):
    for t in range(sequence_length):
        obs[t] = means[states[t]] + stds[states[t]] * np.random.randn()


class HMMDataset(Dataset):
    def __init__(self, num_sequences, sequence_length, trans, means, stds, device=None):
        """
        Args:
            num_sequences (int): Number of sequences in the dataset.
            sequence_length (int): Length of each sequence.
            trans (np.ndarray): Transition matrix of shape (num_states, num_states).
            means (list): List of means for Gaussian emissions for each state.
            stds (list): List of standard deviations for Gaussian emissions for each state.
        """
        self.device = get_device() if device is None else device
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.trans = trans
        self.means = means
        self.stds = stds
        self.num_states = len(self.means)

        self.states = np.zeros(self.sequence_length, dtype=int)
        self.observations = np.zeros(self.sequence_length, dtype=float)

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
        self.observations[:] = 0.0

        # NB uniform categorical prior on starting state.
        self.states[0] = np.random.choice(self.num_states)

        populate_states(self.states, self.num_states, self.trans)

        populate_obs(
            self.observations, self.states, self.means, self.stds, self.sequence_length
        )

        states = torch.tensor(self.states, dtype=torch.long, device=self.device)
        observations = torch.tensor(
            self.observations, dtype=torch.float, device=self.device
        ).unsqueeze(-1)

        logger.debug(f"{states}")
        logger.debug(f"Realized HMM simulation:\n{observations}")

        # NB when called as a batch, will have shape [batch_size, seq_length, 1].
        return observations, states


if __name__ == "__main__":
    num_states = 2
    num_sequences = 256
    sequence_length = 4
    batch_size = 1
    num_layers = 1
    learning_rate = 1.0

    # NB defines true parameters.
    trans = np.array([[1.0, 0.0], [0.0, 1.0]])

    means = [5.0, 10.0]
    stds = [1.0, 1.0]

    dataset = HMMDataset(
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        trans=trans,
        means=means,
        stds=stds,
    )

    dataset_iter = iter(dataset)
    data = next(dataset_iter)

    print(data)
