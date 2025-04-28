import torch
from algos.rnn.utils import get_device

def test_get_device():
    assert get_device("cpu") == torch.device("cpu", index=0)

    # NB default to config defined "native"
    assert (device := get_device()) == torch.device("mps:0")

