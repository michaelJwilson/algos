import torch
from algos.rnn.hmm import HMM
from algos.rnn.utils import get_device
from algos.rnn.config import Config


def test_hmm_autodiff():
    config = Config()
    
    num_states = config.num_states
    batch_size = config.batch_size
    sequence_length = config.sequence_length

    device = get_device()
    
    observations = torch.randint(0, num_states, (batch_size, sequence_length), device=device)

    print(f"\n{observations}")
    
    model = HMM(batch_size, sequence_length, num_states)
    model.zero_grad()

    log_gamma = model(observations)

    loss = log_gamma.sum()

    loss.backward()
    
    for name, param in model.named_parameters():
        print(f"{name}:  {param.grad}")    
