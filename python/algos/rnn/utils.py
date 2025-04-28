import random
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

def get_device():
    # TODO HACK
    device = "cpu"

    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    """
    logger.info("Utilizing the {device} device.")
        
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
