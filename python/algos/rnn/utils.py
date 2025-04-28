import random
import logging

import numpy as np
import torch

from algos.rnn.config import Config

logger = logging.getLogger(__name__)


def get_device(device=None, index=0):
    """
    Returns a torch device according to the priority:
      - keyword argument
      - config definition, one of "native" or named,
        e.g. cpu.
    """
    if device is None:
        if (device := Config().device) == "native":
            if torch.backends.mps.is_available():
                # TODO HACK
                device = "mps"
            elif torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = "cpu"
                
    logger.debug(f"Utilizing {device}:{index} device.")

    return torch.device(f"{device}:{index}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
