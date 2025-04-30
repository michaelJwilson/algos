import yaml
import logging

from pathlib import Path
from pprint import pformat
from types import SimpleNamespace

logger = logging.getLogger(__name__)


class Config:
    def __init__(self, config_path=None):
        config = self.load_config(config_path)

        self.blocks = config.keys()

        for name in self.blocks:
            setattr(self, name, SimpleNamespace(**config.get(name, {})))

    def __repr__(self):
        result = {}

        for block in self.blocks:
            result[block] = pformat(vars(getattr(self, block)), indent=4)

        return (
            "\nConfig(\n" + "\n".join(f"{k}:\n{v}\n" for k, v in result.items()) + ")\n"
        )

    def __getattr__(self, name):
        for block in self.blocks:
            if hasattr(getattr(self, block), name):
                return getattr(getattr(self, block), name)

        msg = f"'Config' object has no attribute '{name}'"

        raise AttributeError(msg)

    def get_config_path(self):
        return Path(__file__).resolve().parent / "config.yaml"

    def load_config(self, fpath=None):
        fpath = self.get_config_path() if fpath is None else fpath

        with Path.open(fpath) as file:
            return yaml.safe_load(file)
