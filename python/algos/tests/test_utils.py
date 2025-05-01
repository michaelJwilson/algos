import torch
from algos.rnn.utils import get_device


def test_get_device():
    assert get_device("cpu").type == torch.device("cpu", index=0).type

    # NB default to config defined "native"
    assert (device := get_device()).type == torch.device("mps:0").type
