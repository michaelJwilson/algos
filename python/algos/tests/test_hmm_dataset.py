import numpy as np
import pytest
import torch
from algos.rnn.hmm_dataset import HMMDataset


@pytest.fixture
def hmm_dataset():
    num_states = 2
    num_sequences = 10
    sequence_length = 5
    jump_rate = 0.5

    return HMMDataset(num_states, num_sequences, sequence_length, jump_rate)


def test_hmm_dataset_item_shape(hmm_dataset):
    obvs, states = hmm_dataset[0]

    assert len(hmm_dataset) == 10

    assert torch.is_tensor(obvs)
    assert obvs.dtype == torch.float

    assert obvs.shape == (5, 1)
    assert states.shape == (5,)

    assert torch.all((states >= 0) & (states < len(hmm_dataset.means)))


def test_hmm_hmm_dataset_transition_matrix(hmm_dataset):
    trans = hmm_dataset.trans
    obvs, states = hmm_dataset[0]

    assert trans.shape == (2, 2)
    assert np.allclose(trans.sum(axis=1), 1.0)

    assert len(states) == hmm_dataset.sequence_length
    assert states[0] in range(hmm_dataset.num_states)

    assert len(obvs) == hmm_dataset.sequence_length
    assert obvs.ndim == 2
