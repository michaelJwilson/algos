import random

import numpy as np
import torch


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Use MPS (GPU)
    else:
        device = torch.device("cpu")  # Fallback to CPU

    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
