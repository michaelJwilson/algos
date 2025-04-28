import pytest
import torch

from algos.rnn.embedding import GaussianEmbedding
from algos.rnn.config import Config


@pytest.fixture
def gaussian_embedding():
    num_states = Config().dataset.num_states
    return GaussianEmbedding(num_states)


def test_initialization(gaussian_embedding):
    assert gaussian_embedding.means.shape == (num_states,), "Means shape is incorrect"
    assert gaussian_embedding.log_vars.shape == (num_states,), "Log vars shape is incorrect"
    assert gaussian_embedding.means.requires_grad, "Means should require gradients"
    assert not gaussian_embedding.log_vars.requires_grad, "Log vars should not require gradients"


def test_forward_pass(gaussian_embedding):
    batch_size, sequence_length, num_features = 4, 10, 1
    output = gaussian_embedding(
        torch.randn(batch_size, sequence_length, num_features)
    )

    assert output.shape == (
        batch_size,
        sequence_length,
        num_states,
    ), "Output shape is incorrect"


def test_device_compatibility(gaussian_embedding):
    assert (
        gaussian_embedding.means.device == gaussian_embedding.device
    ), "Means are not on the correct device"
    assert (
        gaussian_embedding.log_vars.device == gaussian_embedding.device
    ), "Log vars are not on the correct device"

    batch_size, sequence_length, num_features = 4, 10, 1
    output = gaussian_embedding(
        torch.randn(batch_size, sequence_length, num_features).to(gaussian_embedding.device)
    )

    assert output.device == gaussian_embedding.device, "Output is not on the correct device"
