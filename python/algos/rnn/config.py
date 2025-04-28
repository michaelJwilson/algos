import os
from pprint import pformat
from types import SimpleNamespace

import yaml


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

        raise AttributeError(f"'Config' object has no attribute '{name}'")
    
    def get_config_path(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        return os.path.join(script_dir, "config.yaml")

    def load_config(self, fpath=None):
        fpath = self.get_config_path() if fpath is None else fpath

        with open(fpath, "r") as file:
            config = yaml.safe_load(file)

        return config
