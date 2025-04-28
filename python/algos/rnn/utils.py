import random
import logging

import numpy as np
import torch

from algos.rnn.config import Config

logger = logging.getLogger(__name__)

def get_device():
    if device := Config().device == "native":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"            
        else:
            device = "cpu"
            
    logger.info("Utilizing the {device} device.")
        
    return torch.device(device)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
