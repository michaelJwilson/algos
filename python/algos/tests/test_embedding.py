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
def normal_embedding():
    return GaussianEmbedding()


@pytest.fixture
def betabinomial_embedding(config, device):
    coverage = torch.ones(config.batch_size, config.sequence_length, 1, device=device)

    return BetaBinomialEmbedding(coverage)


@pytest.fixture
def embedding(request, config, device):
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


# TODO (cpu, index=0) vs cpu.
@pytest.mark.xfail(raises=AssertionError)
def test_embedding_device(normal_embedding):
    assert normal_embedding.means.device == normal_embedding.device
    assert normal_embedding.log_vars.device == normal_embedding.device
