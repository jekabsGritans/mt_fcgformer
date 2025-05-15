"""
This is an adaptation of a reimplementation of the IRCNN model in PyTorch.

Original Model (Keras Based): https://github.com/gj475/irchracterizationcnn
Pytorch Reimplementation: https://github.com/lycaoduong/FcgFormer
"""

from typing import Any, TypedDict

import numpy as np
import torch
import torch.nn as nn
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema
from mlflow.types.schema import Array
from omegaconf import DictConfig

from models.base_model import BaseModel, NeuralNetworkModule


class InputRow(TypedDict):
    spectrum: np.ndarray  # (input_dim,)
    threshold: float

class OutputRow(TypedDict):
    positive_targets: list[str]
    positive_probabilities: list[float]

class IrCNNModule(NeuralNetworkModule):
    """Neural network architecture for IrCNN"""
    
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, dropout_p: float, pos_weights: list[float] | None = None):
        """
        Initialize IrCNN neural network.
        
        Args:
            input_dim: Input dimension (number of features)
            output_dim: Output dimension (number of classes)
            kernel_size: Kernel size for the convolutional layers (hyperparameter)
            dropout_p: Dropout probability (hyperparameter)
            pos_weights: Positive weights for BCE loss
        """
        super().__init__(input_dim, output_dim, pos_weights)
        
        in_ch = 1  # this is fixed for our repository

        # 1st CNN layer
        self.CNN1 = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=31, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm1d(num_features=31),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.cnn1_size = int(((input_dim - kernel_size + 1 - 2) / 2) + 1)
        
        # 2nd CNN layer
        self.CNN2 = nn.Sequential(
            nn.Conv1d(in_channels=31, out_channels=62, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm1d(num_features=62),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.cnn2_size = int(((self.cnn1_size - kernel_size + 1 - 2) / 2) + 1)

        # Dense layers
        self.DENSE1 = nn.Sequential(
            nn.Linear(in_features=self.cnn2_size * 62, out_features=4927),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        self.DENSE2 = nn.Sequential(
            nn.Linear(in_features=4927, out_features=2785),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        self.DENSE3 = nn.Sequential(
            nn.Linear(in_features=2785, out_features=1574),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        # FCN layer
        self.FCN = nn.Linear(in_features=1574, out_features=output_dim)

    def forward(self, signal):
        x = signal.unsqueeze(dim=1)
        x = self.CNN1(x)
        x = self.CNN2(x)
        x = torch.flatten(x, -2, -1)
        x = torch.unsqueeze(x, dim=1)
        x = self.DENSE1(x)
        x = self.DENSE2(x)
        x = self.DENSE3(x)
        x = self.FCN(x)
        x = torch.squeeze(x, dim=1)
        return x


class IrCNN(BaseModel):

    def init_from_config(self, cfg: DictConfig):

        # TODO: target names and pos weights are null by default in config, in which case computed from dataset

        # Initialize the network
        self.nn = IrCNNModule(cfg.model.input_dim, cfg.model.output_dim, cfg.model.kernel_size, cfg.model.dropout_p)
        self.target_names = cfg.target_names
        
        # Define MLflow schemas
        input_schema = Schema([
            ColSpec(type=Array(DataType.double), name="spectrum"),
            ColSpec(type=DataType.double, name="threshold")
        ]) 

        output_schema = Schema([
            ColSpec(
                type=Array(DataType.string),
                name="positive_targets",
            ),
            ColSpec(
                type=Array(DataType.double),
                name="positive_probabilities"
            ),
        ])

        param_schema = ParamSchema([
            ParamSpec(
                name="threshold",
                dtype=DataType.double,
                default=0.5
            )
        ])

        input_example = (
            [
                {
                    "spectrum": np.zeros(cfg.model.input_dim, dtype=np.float32)
                }
            ],
            {"threshold": 0.5}
        )

        self._signature = ModelSignature(
            inputs=input_schema,
            outputs=output_schema,
            params=param_schema
        )

        self._input_example = input_example

    # MLFlow
    def predict(self, context, model_input: list[InputRow], params: dict[str, Any] | None = None):
        """
        Make predictions with the model.
        
        Args:
            context: MLflow context
            model_input: List of input rows
            params: Additional parameters
            
        Returns:
            List of predictions
        """
        assert self.target_names is not None, "Target names not set."

        threshold = params.get("threshold", 0.5) if params else 0.5
        results = []
        
        for row in model_input:
            spectrum = torch.tensor(row["spectrum"], dtype=torch.float32)

            # preprocess like in evaluation
            spectrum = self.spectrum_eval_transform(spectrum)

            # add batch dim
            spectrum = spectrum.unsqueeze(0)

            # forward pass
            logits = self.nn.forward(spectrum)
            probabilities = torch.sigmoid(logits).squeeze(0).tolist()

            out_probs, out_targets = [], []
            for prob, target in zip(probabilities, self.target_names):
                if prob > threshold:
                    out_probs.append(prob)
                    out_targets.append(target)
            
            results.append({
                "positive_targets": out_targets, 
                "positive_probabilities": out_probs
            })
        
        return results
        
    def load_context(self, context):
        """
        Load model weights from MLflow artifacts.
        
        Args:
            context: MLflow context
        """
        checkpoint_path = context.artifacts["checkpoint"]
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.nn.load_state_dict(state_dict)
        self.nn.eval()