from __future__ import annotations

import os
from re import A
from typing import Callable

import numpy as np
import pandas as pd
import torch
from matplotlib.pylab import float_
from omegaconf import DictConfig
from torch.utils.data import Dataset

from utils.mlflow_utils import download_artifact

# our transforms are just user defined functions
Transform = Callable[[torch.Tensor], torch.Tensor]

"""
df

Required:
    spectrum: float array (already interpolated not XY). 
    wavenumber_interval: (min, max) in cm^-1

Functional groups by name (just give in example, not enforced)
    fg_name: bool

additional targets:
    target: bool or float (int num donors as just float)

if doing augmentations for synthetic samples, 
    augmented: bool # to not use for validation/test

dataset split performed at object init, with a local seed so that the same for test/valid.
"""

#TODO: save head as txt to mlflow. and description of relevant columns. prolly no params

class MLFlowDataset(Dataset):
    """
    Downloads and stores dataset from MLFlow.
    """

    df: pd.DataFrame

    spectra: torch.Tensor # (num_samples,) input is always just spectra. not normalized.

    fg_names: list[str] # names of functional groups
    fg_targets: torch.Tensor # (num_samples, fg) bool
    fg_pos_weights: torch.Tensor # (fg,) float

    aux_bool_names: list[str] # names of auxiliary bool targets
    aux_bool_targets: torch.Tensor # (num_samples, aux_targets) bool
    aux_pos_weights: torch.Tensor # (aux_targets,) float

    aux_float_names: list[str] # names of float targets
    aux_float_targets: torch.Tensor # (num_samples, float_targets) float

    spectrum_transform: Transform | None # random train transforms only applied to spectra

    cfg: DictConfig
    dataset_id: str
    split: str

    def __init__(self, cfg: DictConfig, dataset_id: str, split: str, transform: Transform | None = None):
        """
        Initialize the dataset.
        Args:
            cfg (DictConfig): Configuration object.
            dataset_id (str): MLFlow run ID of the dataset to download.
            split (str): Split of the dataset to download (train, valid, test).
            transform (Transform | None): Transform to apply to the inputs.
        """
        super().__init__()
        self.cfg = cfg
        self.dataset_id = dataset_id
        self.transform = transform
        self.split = split
        
    def load_df(self, df: pd.DataFrame):
        assert "spectrum" in df.columns, "spectrum column not found in dataframe"

        self.df = df

        # spectra
        self.spectra = torch.Tensor(df["spectrum"].tolist(), dtype=torch.float32, device=self.cfg.device) # type: ignore

        # functional groups
        cols = []
        for target in self.fg_names:
            assert target in df.columns, f"{target} column not found in dataframe"
            assert df[target].dtype == bool, f"{target} column is not bool"
            col = torch.tensor(df[target].tolist(), dtype=torch.bool, device=self.cfg.device) # type: ignore
            cols.append(col)

        self.fg_targets = torch.stack(cols, dim=1)

        fg_pos_counts = self.fg_targets.sum(dim=0).tolist()
        fg_pos_weights = self._compute_pos_weights(fg_pos_counts)
        self.fg_pos_weights = torch.tensor(fg_pos_weights, dtype=torch.float32, device=self.cfg.device) # type: ignore

        # auxiliary bool
        cols = []
        for target in self.aux_bool_names:
            assert target in df.columns, f"{target} column not found in dataframe"
            assert df[target].dtype == bool, f"{target} column is not bool"
            col = torch.tensor(df[target].tolist(), dtype=torch.bool, device=self.cfg.device) # type: ignore
            cols.append(col)

        self.aux_bool_targets = torch.stack(cols, dim=1)

        aux_bool_pos_counts = self.aux_bool_targets.sum(dim=0).tolist()
        aux_bool_pos_weights = self._compute_pos_weights(aux_bool_pos_counts)
        self.aux_pos_weights = torch.tensor(aux_bool_pos_weights, dtype=torch.float32, device=self.cfg.device) # type: ignore

        # auxiliary float
        cols = []
        for target in self.aux_float_names:
            assert target in df.columns, f"{target} column not found in dataframe"
            assert df[target].dtype == float, f"{target} column is not float"
            col = torch.tensor(df[target].tolist(), dtype=torch.float32, device=self.cfg.device)
            cols.append(col)

        self.aux_float_targets = torch.stack(cols, dim=1)

    def download(self):
        df_path = download_artifact(self.cfg, self.dataset_id, f"{self.split}_df.pkl")
        self.df = pd.read_pickle(df_path)
        self.load_df(self.df)

    def _compute_pos_weights(self, pos_counts: list[int]) -> list[float]:
        """
        Compute positive weights for each target class to address class imbalance.
        
        The positive weight for each class is calculated as (total_samples - pos_count) / pos_count,
        which gives higher weights to underrepresented positive examples.
        
        Args:
            pos_counts (list[int]): Number of positive samples for each target class
            
        Returns:
            list[float]: Positive weights for each target class
        """
        total_samples = self.spectra.shape[0]
        pos_weights = []
        
        for pos_count in pos_counts:
            # Avoid division by zero
            assert pos_count > 0, "Positive count cannot be zero"
            weight = (total_samples - pos_count) / pos_count
            pos_weights.append(weight)
        
        return pos_weights
    
    def to(self, device: torch.device) -> MLFlowDataset:
        """
        Move the dataset to the specified device.
        :param device: Device to move the dataset to
        """
        self.sprctra = self.spectra.to(device)
        self.targets = {name: target.to(device) for name, target in self.targets.items()}       

        return self

    def __len__(self):
        """
        Get the number of samples in the dataset.
        :return: Number of samples
        """
        return self.spectra.shape[0]
    
    def __getitem__(self, index) -> dict:
        """
        Get a sample from the dataset.
        :param index: Index of the sample
        :return: output dictionary
        """

        out = {}

        spectrum = self.spectra[index]
        if self.spectrum_transform is not None:
            spectrum = self.spectrum_transform(spectrum)

        out["spectrum"] = spectrum
        out["fg_targets"] = self.fg_targets[index]
        out["aux_bool_targets"] = self.aux_bool_targets[index]
        out["aux_float_targets"] = self.aux_float_targets[index]

        return out





