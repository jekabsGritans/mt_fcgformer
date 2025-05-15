from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from mlflow.models import ModelSignature
from mlflow.pyfunc import PythonModel  # type: ignore
from omegaconf import DictConfig, OmegaConf


# Define neural network module for the model architecture
class NeuralNetworkModule(nn.Module, ABC):
    """Base neural network architecture for multilabel classification"""
    
    def __init__(self, input_dim: int, output_dim: int, pos_weights: list[float] | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.set_pos_weights(pos_weights)
    
    def set_pos_weights(self, pos_weights: list[float] | None = None):
        """Set weights for BCE loss"""
        if pos_weights is None:
            pos_weights = [1.0] * self.output_dim
            
        torch_pos_weights = torch.tensor(pos_weights, dtype=torch.float32)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch_pos_weights)
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network"""
        pass
    
    def compute_loss(self, logits, targets):
        return self.loss_fn(logits, targets.float())

    def step(self, batch: dict) -> dict:
        """
        Perform a single forward pass on a batch and return the loss.
        """
        x = batch["inputs"]
        y = batch["target"]
        logits = self.forward(x)
        loss = self.compute_loss(logits, y)
        
        return {
            "logits": logits,
            "loss": loss,
        }


class BaseModel(PythonModel, ABC):
    """
    Base class for multilabel classification models with
    - 1d input (IR spectra)
    - 1d output (multilabel classification)
    - weighted BCE loss.
    
    This class serves as the MLflow PyFuncModel wrapper around a neural network.
    """

    neural_net: NeuralNetworkModule
    target_names: list[str]

    _signature: ModelSignature
    _input_example: Any
    
    def __init__(self, model_cfg: DictConfig | None = None):
        super().__init__()

        if model_cfg is not None:
            self.init_from_config(model_cfg)
    
    @abstractmethod
    def predict(self, context, model_input, params=None):
        """
        MLflow predict method to be implemented by subclasses
        """

    @abstractmethod
    def init_from_config(self, cfg: DictConfig):
        """
        Initialize the model.
        """

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load the model checkpoint.
        """
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        self.neural_net.load_state_dict(checkpoint_data)
        self.neural_net.eval()
    
    def load_context(self, context):
        """
        Called by MLflow to load the model context.
        """
        cfg_path = context.artifacts.get("model_config")
        cfg = OmegaConf.load(cfg_path)
        self.init_from_config(cfg) # type: ignore

        checkpoint_path = context.artifacts.get("model_checkpoint")
        self.load_checkpoint(checkpoint_path)
