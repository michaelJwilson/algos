import os
import yaml

def get_config_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    return os.path.join(script_dir, "config.yaml")
    
def load_config():
    """
    Load configuration parameters from a YAML file.

    Args:
        config_path (str): Path to the configuration YAML file.

    Returns:
        dict: Dictionary containing configuration parameters.
    """
    fpath = get_config_path()
    
    with open(fpath, "r") as file:
        config = yaml.safe_load(file)

    return config
