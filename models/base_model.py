from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from mlflow.models import ModelSignature
from mlflow.pyfunc import PythonModel  # type: ignore
from omegaconf import DictConfig

from utils.transforms import Transform


# Define neural network module for the model architecture
class NeuralNetworkModule(nn.Module, ABC):
    """Base neural network architecture for multilabel classification"""
    
    def __init__(self, spectrum_dim: int, fg_target_dim: int, aux_bool_target_dim: int, aux_float_target_dim: int,
                 fg_pos_weights: list[float] | None = None, aux_pos_weights: list[float] | None = None,
                 fg_loss_weight: float = 1.0, aux_bool_loss_weight: float = 0.5, aux_float_loss_weight: float = 0.1):
        super().__init__()
        self.spectrum_dim = spectrum_dim
        self.fg_target_dim = fg_target_dim
        self.aux_bool_target_dim = aux_bool_target_dim
        self.aux_float_target_dim = aux_float_target_dim
        self.setup_loss(fg_pos_weights, aux_pos_weights)
        self.fg_loss_weight = fg_loss_weight
        self.aux_bool_loss_weight = aux_bool_loss_weight
        self.aux_float_loss_weight = aux_float_loss_weight
    
    def setup_loss(self, fg_pos_weights: list[float] | None = None, aux_pos_weights: list[float] | None = None):

        # calssification of fg targets
        if fg_pos_weights is None:
            fg_pos_weights = [1.0] * self.fg_target_dim
            
        torch_fg_pos_weights = torch.tensor(fg_pos_weights, dtype=torch.float32)
        self.fg_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch_fg_pos_weights)


        # classification of auxiliary bool targets
        if self.aux_bool_target_dim > 0:
            if aux_pos_weights is None:
                aux_pos_weights = [1.0] * self.aux_bool_target_dim
            torch_aux_pos_weights = torch.tensor(aux_pos_weights, dtype=torch.float32)
            self.aux_bool_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch_aux_pos_weights)

        # regression of auxiliary float targets
        if self.aux_float_target_dim > 0:
            self.aux_float_loss_fn = nn.MSELoss()

    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through the neural network"""
        pass
    
    def step(self, batch: dict) -> dict:
        """
        Perform a single forward pass on a batch and return the loss.
        """

        spectrum = batch["spectrum"]

        preds = self.forward(spectrum)

        out = {}

        fg_targets = batch["fg_targets"]
        fg_logits = preds["fg_logits"]
        fg_loss = self.fg_loss_fn(fg_logits, fg_targets.float())
        out["fg_logits"] = fg_logits
        out["fg_loss"] = fg_loss

        if self.aux_bool_target_dim > 0:
            aux_bool_targets = batch["aux_bool_targets"]
            aux_bool_logits = preds["aux_bool_logits"]
            aux_bool_loss = self.aux_bool_loss_fn(aux_bool_logits, aux_bool_targets.float())
            out["aux_bool_logits"] = aux_bool_logits
            out["aux_bool_loss"] = aux_bool_loss
        else:
            aux_bool_loss = 0.0
        
        if self.aux_float_target_dim > 0:
            aux_float_targets = batch["aux_float_targets"]
            aux_float_preds = preds["aux_float_preds"]
            aux_float_loss = self.aux_float_loss_fn(aux_float_preds, aux_float_targets.float())
            out["aux_float_preds"] = aux_float_preds
            out["aux_float_loss"] = aux_float_loss
        else:
            aux_float_loss = 0.0

        total_loss = (self.fg_loss_weight * fg_loss +
                self.aux_bool_loss_weight * aux_bool_loss +
                self.aux_float_loss_weight * aux_float_loss)
        
        out["loss"] = total_loss
       
        return out

class BaseModel(PythonModel, ABC):
    """
    Base class for multilabel classification models with
    - 1d input (IR spectra)
    - 1d output (multilabel classification)
    - weighted BCE loss.
    
    This class serves as the MLflow PyFuncModel wrapper around a neural network.
    """

    nn: NeuralNetworkModule
    fg_names: list[str]

    _signature: ModelSignature
    _input_example: Any

    spectrum_eval_transform: Transform
    
    def __init__(self, cfg: DictConfig | None = None):
        super().__init__()

        if cfg is not None:
            self.init_from_config(cfg)
        
    
    @abstractmethod
    def predict(self, context, model_input: pd.DataFrame, params: dict | None=None):
        """
        MLflow predict method to be implemented by subclasses
        Only outputs fg predictions, not auxiliary
        """

    @abstractmethod
    def init_from_config(self, cfg: DictConfig):
        """
        Initialize the model.
        """

    def set_fg_names(self, fg_names: list[str]):
        """
        Set the functional group names for the model.
        """
        self.fg_names = fg_names

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load the model checkpoint.
        """
        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        self.nn.load_state_dict(checkpoint_data)
        self.nn.eval()