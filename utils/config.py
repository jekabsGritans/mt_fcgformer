"""
Global configuration module that exposes the Hydra config through a function.
"""
import os
from threading import Lock

from omegaconf import DictConfig

# Global variable for config (private to this module)
_config = None
_config_lock = Lock()

def get_config() -> DictConfig:
    """
    Get the global configuration object.
    
    Returns:
        The current Hydra config object
    
    Raises:
        RuntimeError: If config hasn't been set yet
    """
    if _config is None:
        raise RuntimeError("Config not initialized. Call set_config() first.")
    return _config

def set_config(config):
    """
    Set the global configuration object.
    
    Args:
        config: The Hydra config object
    """
    global _config
    with _config_lock:
        _config = config