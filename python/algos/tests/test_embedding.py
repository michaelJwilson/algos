import pytest
import torch
from algos.rnn.embedding import GaussianEmbedding
from algos.rnn.config import Config


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def normal_embedding():
    return GaussianEmbedding()


def test_initialization(normal_embedding):
    num_states = normal_embedding.num_states

    assert normal_embedding.means.shape == (num_states,), "Means shape is incorrect"
    assert normal_embedding.log_vars.shape == (
        num_states,
    ), "Log vars shape is incorrect"
    assert normal_embedding.means.requires_grad, "Means should require gradients"
    assert (
        normal_embedding.log_vars.requires_grad
    ), "Log vars should not require gradients"


def test_embedding_forward(config, normal_embedding):
    result = normal_embedding(
        torch.randn(config.batch_size, config.sequence_length, 1).to(
            normal_embedding.device
        )
    )

    assert result.shape == (
        config.batch_size,
        config.sequence_length,
        config.num_states,
    )


def test_embedding_device(normal_embedding):
    assert normal_embedding.means.device == normal_embedding.device
    assert normal_embedding.log_vars.device == normal_embedding.device
