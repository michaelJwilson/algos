import pytest
import torch
from algos.rnn.utils import get_device
from algos.rnn.config import Config
from algos.rnn.embedding import (
    BetaBinomialEmbedding,
    GaussianEmbedding,
    NegativeBinomialEmbedding,
)


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def device():
    return get_device()


@pytest.fixture
def embedding(request, config, device):
    # NB see https://pytorch.org/docs/stable/generated/torch.ones.html
    coverage = torch.ones(config.batch_size, config.sequence_length, 1, device=device)

    if request.param == "normal":
        return GaussianEmbedding()
    elif request.param == "nbinomial":
        return NegativeBinomialEmbedding()
    elif request.param == "betabinomial":
        return BetaBinomialEmbedding(coverage)
    else:
        raise ValueError(f"{request.param} embedding is not available as a fixture.")


@pytest.mark.parametrize(
    "embedding", ["normal", "betabinomial", "nbinomial"], indirect=True
)
def test_embedding_init(embedding):
    for name, param in embedding.named_parameters():
        # NB guard clause to e.g. remove embedding.coverage; also drops normal.stds
        if param.requires_grad:
            assert param.shape == torch.Size([embedding.num_states])
            assert param.device.type == embedding.device.type


@pytest.mark.parametrize(
    "embedding", ["normal", "betabinomial", "nbinomial"], indirect=True
)
def test_embedding_forward(config, device, embedding):
    # NB see https://pytorch.org/docs/stable/generated/torch.randint.html
    data = torch.randint(
        0,
        10,
        size=(config.batch_size, config.sequence_length, 1),
        dtype=torch.int,
        device=device,
    )

    result = embedding.forward(data)

    assert result.shape == (
        config.batch_size,
        config.sequence_length,
        config.num_states,
    )
