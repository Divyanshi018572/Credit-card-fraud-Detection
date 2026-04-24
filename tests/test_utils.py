import os
from src.utils.config_loader import load_config

def test_load_config():
    """Verify that config can be loaded and has expected top-level keys."""
    config = load_config("configs/config.yaml")
    assert isinstance(config, dict)
    assert "models" in config
    assert "preprocessing" in config
    assert "business_logic" in config

def test_config_values():
    """Verify specific config values are correctly loaded."""
    config = load_config("configs/config.yaml")
    assert config["preprocessing"]["test_size"] == 0.2
    assert config["business_logic"]["risk_tiers"]["critical"] == 0.8
