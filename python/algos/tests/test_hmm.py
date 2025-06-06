import pytest
from algos.rnn.config import Config
from algos.rnn.hmm import HMM
from algos.rnn.hmm_dataset import HMMDataset
from algos.rnn.utils import get_device, set_precision, set_seed
from torch.utils.data import DataLoader

# TODO
set_seed(42)
set_precision()

config = Config()


@pytest.fixture
def hmm_dataset():
    return HMMDataset(
        num_sequences=config.num_sequences,
        sequence_length=config.sequence_length,
        jump_rate=config.jump_rate,
        means=config.means,
        stds=config.stds,
    )


def test_hmm(hmm_dataset):
    # num_workers=1
    dataloader = DataLoader(hmm_dataset, batch_size=config.batch_size, shuffle=False)
    obvs, states = next(iter(dataloader))

    device = get_device()
    model = HMM(config.batch_size, config.sequence_length, config.num_states, device)

    print(f"HMM model summary:\n{model}")

    # NB forward model is lnP to match CrossEntropyLoss()
    estimate = model.forward(obvs.to(device))
