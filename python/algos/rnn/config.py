import os
import yaml


class Config:
    def __init__(self, config_path):
        """
        Initialize the Config class by loading parameters from a YAML file.

        Args:
            config_path (str): Path to the configuration YAML file.
        """
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Partition the configuration into namespaces
        self.dataset = SimpleNamespace(
            **{
                "num_states": config.get("num_states"),
                "num_sequences": config.get("num_sequences"),
                "sequence_length": config.get("sequence_length"),
            }
        )

        self.training = SimpleNamespace(
            **{
                "batch_size": config.get("batch_size"),
                "num_layers": config.get("num_layers"),
                "learning_rate": config.get("learning_rate"),
                "num_epochs": config.get("num_epochs"),
            }
        )

    def __repr__(self):
        return f"Config(dataset={self.dataset}, training={self.training})"

    def get_config_path():
        script_dir = os.path.dirname(os.path.abspath(__file__))

        return os.path.join(script_dir, "config.yaml")

    def load_config(fpath=None):
        fpath = get_config_path() if fpath is None else fpath

        with open(fpath, "r") as file:
            config = yaml.safe_load(file)

        return config
