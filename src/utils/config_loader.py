import yaml
from pathlib import Path

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Loads a YAML configuration file.
    
    Args:
        config_path (str): Path to the yaml file.
        
    Returns:
        dict: Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(path, "r") as f:
        return yaml.safe_load(f)
