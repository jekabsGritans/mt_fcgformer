"""
This is an adaptation of a reimplementation of the IRCNN model in PyTorch.

Original Model (Keras Based): https://github.com/gj475/irchracterizationcnn
Pytorch Reimplementation: https://github.com/lycaoduong/FcgFormer
"""

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema
from mlflow.types.schema import AnyType, Array
from omegaconf import DictConfig

from models.base_model import BaseModel, NeuralNetworkModule
from utils.misc import interpolate
from utils.transforms import Compose


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

        self.spectrum_eval_transform = Compose.from_hydra(cfg.eval_transforms)

        # Initialize the network
        self.nn = IrCNNModule(cfg.model.input_dim, cfg.model.output_dim, cfg.model.kernel_size, cfg.model.dropout_p)
        self.target_names = cfg.target_names
        
        # Input is only spectrum.
        input_schema = Schema([
            ColSpec(Array(DataType.double), name="spectrum_x"),
            ColSpec(Array(DataType.double), name="spectrum_y"),
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
            ColSpec(type=Array(DataType.double), name="positive_probabilities"),

            # interpret as list of list of tuples (min, max, val) where min,max wavenums and val between 0 and 1
            ColSpec(type=AnyType(), name="attention"), # [(400, 4000, 0.0), ...]
        ])

        # Batched input of spectra
        a = np.zeros((100, ), dtype=np.float32).tolist()  # Example input for the schema
        
        input_example = pd.DataFrame({"spectrum_x": [a], "spectrum_y": [a]})  # type: ignore

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
    def predict(self, context, model_input: pd.DataFrame, params: dict | None = None) -> list[dict]:
        """ Make predictions with the model. """

        assert self.target_names is not None, "Target names not set."

        threshold = params.get("threshold", 0.5) if params else 0.5
        results = []

        spectra_x = np.stack(model_input["spectrum_x"].to_numpy()) # type: ignore
        spectra_y = np.stack(model_input["spectrum_y"].to_numpy()) # type: ignore

        self.nn.eval()

        for spectrum_x, spectrum_y in zip(spectra_x, spectra_y):
            # interpolate
            spectrum = interpolate(x=spectrum_x, y=spectrum_y, min_x=400, max_x=4000, num_points=3600)

            # preprocess like in evaluation

            spectrum = torch.from_numpy(spectrum).float()
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

            dummy_attention = []
            for _ in range(len(out_targets)):
                dummy_regions = self.generate_random_non_overlapping_intervals(k=5, min_val=400, max_val=4000, min_interval_length=10)
                att = [(start, end, random.random()) for start, end in dummy_regions]  # Dummy attention values
                dummy_attention.append(att)

            results.append({
                "positive_targets": out_targets, 
                "positive_probabilities": out_probs,
                "attention": dummy_attention,
            })
        
        return results


    def generate_random_non_overlapping_intervals(self, k, min_val, max_val, min_interval_length=1, max_attempts_per_interval=100):
        """
        Generates at most k random non-overlapping intervals within a specified range.

        Intervals are considered non-overlapping if their closed intervals do not intersect.
        For example:
        - [1, 5] and [5, 10] are considered overlapping because they share the point 5.
        - [1, 4] and [5, 10] are non-overlapping.

        Args:
            k (int): The maximum number of intervals to generate. The function will try
                    to generate up to k intervals, but may return fewer if it cannot
                    find enough non-overlapping ones within the given constraints.
            min_val (int): The minimum possible value for any point (start or end) in an interval.
            max_val (int): The maximum possible value for any point (start or end) in an interval.
            min_interval_length (int, optional): The minimum length of a generated interval.
                                                Defaults to 1. Must be a positive integer.
            max_attempts_per_interval (int, optional): The maximum number of attempts to find
                                                    a non-overlapping interval for each
                                                    desired interval. This prevents infinite
                                                    loops in scenarios where space is limited.
                                                    Defaults to 100.

        Returns:
            list: A list of tuples, where each tuple (start, end) represents a non-overlapping interval.
                The intervals in the returned list are sorted by their start times for consistency.
        """
        # --- Input Validation ---
        if min_val >= max_val:
            print("Error: min_val must be strictly less than max_val.")
            return []
        if min_interval_length <= 0:
            print("Error: min_interval_length must be a positive integer.")
            return []
        if min_interval_length > (max_val - min_val):
            print("Warning: min_interval_length is greater than the total available range. No intervals can be generated.")
            return []
        if k <= 0:
            print("Warning: k must be a positive integer. No intervals will be generated.")
            return []

        intervals = [] # This list will store the successfully generated non-overlapping intervals
        
        # Attempt to generate up to 'k' intervals
        for _ in range(k):
            found_non_overlapping = False
            # Try 'max_attempts_per_interval' times to find a suitable interval
            for _attempt in range(max_attempts_per_interval):
                # Calculate the effective maximum possible start point for a new interval.
                # This ensures that even an interval of 'min_interval_length' can fit
                # entirely within the [min_val, max_val] range.
                effective_max_start = max_val - min_interval_length

                # If there's no longer enough space to even fit a minimum length interval,
                # break out of the attempts loop and the outer loop (no more intervals can be added).
                if effective_max_start < min_val:
                    break 

                # Generate a random start point for the new interval within the valid range.
                start = random.randint(min_val, effective_max_start)
                
                # Generate a random end point for the new interval.
                # It must be at least 'start + min_interval_length' and at most 'max_val'.
                end = random.randint(start + min_interval_length, max_val)

                new_interval = (start, end)

                # --- Overlap Check ---
                # Assume no overlap initially
                overlaps = False
                # Iterate through all already generated intervals to check for overlap
                for existing_interval in intervals:
                    s_exist, e_exist = existing_interval # Unpack existing interval (start, end)
                    s_new, e_new = new_interval         # Unpack new candidate interval (start, end)
                    
                    # Condition for overlap between two closed intervals [s1, e1] and [s2, e2]:
                    # They overlap if (s1 <= e2) AND (s2 <= e1).
                    # This correctly identifies overlap even if they just touch at an endpoint (e.g., [1,5] and [5,10]).
                    if s_new <= e_exist and s_exist <= e_new:
                        overlaps = True
                        break # Found an overlap, no need to check further existing intervals
                
                # If no overlap was found with any existing intervals, add the new interval
                if not overlaps:
                    intervals.append(new_interval)
                    found_non_overlapping = True # Mark that we successfully found an interval
                    break # Break from the attempts loop, move to the next desired interval
            
            # If after 'max_attempts_per_interval' attempts, a non-overlapping interval
            # could not be found, stop trying to add more intervals. This prevents
            # the function from getting stuck in an infinite loop if the space is too dense.
            if not found_non_overlapping:
                break

        # Sort the generated intervals by their start times. This makes the output consistent
        # and easier to work with.
        intervals.sort()
        return intervals