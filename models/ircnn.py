"""
This is an adaptation of a reimplementation of the IRCNN model in PyTorch.

Original Model (Keras Based): https://github.com/gj475/irchracterizationcnn
Pytorch Reimplementation: https://github.com/lycaoduong/FcgFormer
"""

from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn
from mlflow.models import ModelSignature
from mlflow.types import (ColSpec, DataType, ParamSchema, ParamSpec, Schema,
                          TensorSpec)
from mlflow.types.schema import Array
from omegaconf import DictConfig

from models.base_model import BaseModel, NeuralNetworkModule


class InputRow(TypedDict):
    spectrum: np.ndarray  # (input_dim,)

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

        # Initialize the network
        self.nn = IrCNNModule(cfg.model.input_dim, cfg.model.output_dim, cfg.model.kernel_size, cfg.model.dropout_p)
        self.target_names = cfg.target_names
        
        # Input is only spectrum.
        input_schema = Schema([
            TensorSpec(np.dtype(np.float32), shape=(-1, cfg.model.input_dim))
        ])

        # Known labels are params

        ## Standard params
        params = [ParamSpec(name="threshold", dtype=DataType.double, default=0.5)]

        ## Known targets 
        ## None by default == could be true or false, so don't fix
        known_targets = [
            ParamSpec(name=target, dtype=DataType.boolean, default=None) for target in cfg.target_names
        ] if cfg.target_names else []

        ## Also bool. False by default, e.g. "hydrogen_bonding"
        flags = []

        param_schema = ParamSchema(params + known_targets + flags)

        # Output is a list of positive targets and their probabilities
        output_schema = Schema([
            ColSpec(type=Array(DataType.string), name="positive_targets"),
            ColSpec(type=Array(DataType.double), name="positive_probabilities")
        ])

        # Batched input of spectra
        input_example = np.zeros((1, cfg.model.input_dim), dtype=np.float32)

        self._signature = ModelSignature(
            inputs=input_schema,
            outputs=output_schema,
            params=param_schema
        )

        self._input_example = input_example

        self._description = f"""
        ## Input:
        -  1D array of shape (-1, {cfg.model.input_dim}) representing the IR spectrum. For a single spectrum, use shape (1, {cfg.model.input_dim}).

        ## Parameters:
        - threshold: float, default=0.5. Above this threshold, the target is considered positive.
        for each target:
            - target_name: bool, default=None. If set, the prediction for this target is fixed.

        ## Output:
        - positive_targets: list of strings, names of the targets predicted to be positive.
        - positive_probabilities: list of floats, probabilities for each target in positive_targets.
        """

    # MLFlow
    def predict(self, context, model_input: np.ndarray, params: dict | None = None) -> list[dict]:
        """ Make predictions with the model. """

        assert self.target_names is not None, "Target names not set."

        threshold = params.get("threshold", 0.5) if params else 0.5
        results = []

        for spectrum in model_input:
            # preprocess like in evaluation
            spectrum = self.spectrum_eval_transform(spectrum)

            # add batch dim
            spectrum = spectrum.unsqueeze(0)

            # forward pass
            logits = self.nn.forward(spectrum)
            probabilities = torch.sigmoid(logits).squeeze(0).tolist()

            # apply known targets
            if params is not None:
                for target in self.target_names:
                    if target in params:
                        if params[target] is not None:
                            probabilities[self.target_names.index(target)] = 1.0 if params[target] else 0.0

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