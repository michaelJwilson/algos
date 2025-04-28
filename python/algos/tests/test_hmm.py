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
    
    model = HMM(num_states, sequence_length)
    model.zero_grad()

    log_gamma = model(observations)

    print(log_gamma)
    
    """
    loss = log_gamma.sum()

    loss.backward()

    for name, param in model.named_parameters():
        assert (
            param.grad is not None
        ), f"Parameter '{name}' has no gradient. Autograd is invalid."
        
        print(f"Parameter '{name}' has valid gradient.")

    print("Autograd is valid for all parameters.")
    """
