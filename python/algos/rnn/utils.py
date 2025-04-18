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
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch GPU
        torch.cuda.manual_seed_all(seed)  # All GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-tuning for reproducibility
