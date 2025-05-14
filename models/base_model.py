from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from mlflow.pyfunc import PythonModel  # type: ignore
from mlflow.types import Schema


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


class BaseModel(PythonModel, ABC):
    """
    Base class for multilabel classification models with
    - 1d input (IR spectra)
    - 1d output (multilabel classification)
    - weighted BCE loss.
    
    This class serves as the MLflow PyFuncModel wrapper around a neural network.
    """
    
    def __init__(self, 
                 neural_net: NeuralNetworkModule, 
                 target_names: list[str],
                 input_schema: Schema,
                 output_schema: Schema,
                 input_example: Any):
        """
        Initialize BaseModel.
        
        Args:
            neural_net: Neural network module
            target_names: Names of the target labels
            input_schema: MLflow schema for input
            output_schema: MLflow schema for output
            input_example: Example input for MLflow model signature
        """
        super().__init__()
        self.neural_net = neural_net
        self.target_names = target_names
        self._input_schema = input_schema
        self._output_schema = output_schema
        self._input_example = input_example
    
    def step(self, batch: dict) -> dict:
        """
        Perform a single forward pass on a batch and return the loss.
        """
        x = batch["inputs"]
        y = batch["target"]
        logits = self.neural_net.forward(x)
        loss = self.neural_net.compute_loss(logits, y)
        
        return {
            "logits": logits,
            "loss": loss,
        }
    
    @abstractmethod
    def predict(self, context, model_input, params=None):
        """
        MLflow predict method to be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def load_context(self, context):
        """
        MLflow load_context method to be implemented by subclasses
        """
        pass