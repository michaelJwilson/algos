import torch
from algos.rnn.hmm import HMM


def test_hmm_autodiff():
    num_states = 3
    sequence_length = 5
    batch_size = 2

    observations = torch.randint(0, num_states, (batch_size, sequence_length))

    print(f"\n{observations}")
    
    model = HMM(num_states, sequence_length)
    model.zero_grad()
    """
    log_gamma = model(observations)
    
    loss = log_gamma.sum()

    loss.backward()

    for name, param in model.named_parameters():
        assert (
            param.grad is not None
        ), f"Parameter '{name}' has no gradient. Autograd is invalid."
        
        print(f"Parameter '{name}' has valid gradient.")

    print("Autograd is valid for all parameters.")
    """
