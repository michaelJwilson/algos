import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)

class HMMDataset(Dataset):
    def __init__(self, num_sequences, sequence_length, trans, means, stds):
        """
        Args:
            num_sequences (int): Number of sequences in the dataset.
            sequence_length (int): Length of each sequence.
            trans (np.ndarray): Transition matrix of shape (num_states, num_states).
            means (list): List of means for Gaussian emissions for each state.
            stds (list): List of standard deviations for Gaussian emissions for each state.
        """
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.trans = trans
        self.means = means
        self.stds = stds
        self.num_states = trans.shape[0]

        logger.info(
            f"Generating HMMDataset with true parameters:\nM={self.means}\nT=\n{self.trans}"
        )

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Generate a single HMM sequence on the fly.
        """
        states = np.zeros(self.sequence_length, dtype=int)
        observations = np.zeros(self.sequence_length, dtype=float)

        states[0] = np.random.choice(self.num_states)

        for t in range(1, self.sequence_length):
            states[t] = np.random.choice(self.num_states, p=self.trans[states[t - 1]])

        for t in range(self.sequence_length):
            observations[t] = np.random.normal(
                self.means[states[t]], self.stds[states[t]]
            )

        states = torch.tensor(states, dtype=torch.long, device=device)
        observations = torch.tensor(
            observations, dtype=torch.float, device=device
        ).unsqueeze(-1)

        # NB when called as a batch, will have shape [batch_size, seq_length, 1].
        return observations, states