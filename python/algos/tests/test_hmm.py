import torch
import pytest
import logging

from algos.rnn.hmm import HMM
from algos.rnn.utils import get_device, set_seed
from algos.rnn.config import Config
from algos.rnn.hmm_dataset import HMMDataset
from torch.utils.data import DataLoader


def test_hmm():
    set_seed(42)

    config = Config()
    dataset = HMMDataset(
        num_sequences=config.num_sequences,
        sequence_length=config.sequence_length,
        jump_rate=config.jump_rate,
        means=config.means,
        stds=config.stds,
    )

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    obvs, states = next(iter(dataloader))

    model = HMM(config.batch_size, config.sequence_length, config.num_states)

    print(f"RNN model summary:\n{model}")

    # NB forward model is lnP to match CrossEntropyLoss()                                                                                                                           
    estimate = model.forward(obvs)

    print(estimate)
