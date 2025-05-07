"""
Global configuration module that exposes the Hydra config as a module-level variable.
"""
from threading import Lock

from omegaconf import DictConfig

# Module-level config variable that will be set by set_cfg
# Initially None, will be updated when set_cfg is called
cfg: DictConfig = None # type: ignore

# Lock for thread safety when updating the config
_cfg_lock = Lock()

def set_cfg(new_cfg):
    """
    Set the global configuration.
    This should be called at the beginning of the program
    with the Hydra config object.
    
    Args:
        new_cfg: Hydra config object
    """
    global cfg
    with _cfg_lock:
        cfg = new_cfg