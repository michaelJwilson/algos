import logging
import random

import numpy as np
import torch

from algos.rnn.config import Config

logger = logging.getLogger(__name__)
config = Config()


def set_precision():
    # NB mps has float only.
    torch.set_default_dtype(torch.float32)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device=None, index=0):
    """
    Returns a torch device according to the priority:
      - keyword argument
      - config definition, one of "native" or named,
        e.g. cpu.
    """
    if (device is None) and (device := Config().device) == "native":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = torch.cuda.current_device()
        elif torch.xpu.is_available():
            device = "xpu"
        else:
            device = "cpu"

    logger.debug(f"Utilizing {device}:{index} device.")

    return torch.device(f"{device}:{index}")


def torch_compile(model):
    if config.compile:
        return torch.compile(model)

    return model


def logmatexp(transfer, ln_probs):
    max_ln_probs = torch.max(ln_probs)

    return max_ln_probs + torch.log(torch.exp(ln_probs - max_ln_probs) @ transfer)
