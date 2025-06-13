from __future__ import annotations

import ast
import os
from calendar import c
from typing import Callable

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import (ConcatDataset, DataLoader, Dataset,
                              WeightedRandomSampler)
from zmq import has

from utils.mlflow_utils import download_artifact

# our transforms are just user defined functions
Transform = Callable[[torch.Tensor], torch.Tensor]

class MLFlowDataset(Dataset):
    """
    Downloads and stores dataset from MLFlow.
    """

    df: pd.DataFrame

    spectra: torch.Tensor # (num_samples,) input is always just spectra. not normalized.

    fg_names: list[str] # names of functional groups
    fg_targets: torch.Tensor # (num_samples, fg) bool
    fg_pos_weights: list[float] # (fg,) float

    aux_bool_names: list[str] # names of auxiliary bool targets
    aux_bool_targets: torch.Tensor | None # (num_samples, aux_targets) bool
    aux_pos_weights: list[float] | None # (aux_targets,) float

    aux_float_names: list[str] # names of float targets
    aux_float_targets: torch.Tensor | None # (num_samples, float_targets) float

    spectrum_transform: Transform | None # random train transforms only applied to spectra

    device: torch.device | str

    def __init__(self, device: torch.device | str,
                 fg_names: list[str], aux_bool_names: list[str], aux_float_names: list[str],
                 spectrum_transform: Transform | None = None):
        super().__init__()
        self.device = device

        assert len(fg_names) > 0, "fg_names must not be empty"
        self.fg_names = fg_names
        self.aux_bool_names = aux_bool_names
        self.aux_float_names = aux_float_names
        
        self.spectrum_transform = spectrum_transform

    def load_df(self, df: pd.DataFrame):
        assert "spectrum" in df.columns, "spectrum column not found in dataframe"

        self.df = df

        # spectra
        self.spectra = torch.tensor(np.stack(df["spectrum"]), dtype=torch.float32, device=self.device) # type: ignore

        # functional groups
        cols = []
        for target in self.fg_names:
            assert target in df.columns, f"{target} column not found in dataframe"
            assert df[target].dtype == bool, f"{target} column is not bool"
            col = torch.tensor(df[target].to_numpy(), dtype=torch.bool, device=self.device) # type: ignore
            cols.append(col)

        self.fg_targets = torch.stack(cols, dim=1)

        fg_pos_counts = self.fg_targets.sum(dim=0).tolist()
        self.fg_pos_weights = self._compute_pos_weights(fg_pos_counts)

        # auxiliary bool
        self.aux_pos_weights = None
        self.aux_bool_targets = None
        if len(self.aux_bool_names) > 0:
            cols = []
            for target in self.aux_bool_names:
                assert target in df.columns, f"{target} column not found in dataframe"
                assert df[target].dtype == bool, f"{target} column is not bool"
                col = torch.tensor(df[target].to_numpy(), dtype=torch.bool, device=self.device) # type: ignore
                cols.append(col)

            self.aux_bool_targets = torch.stack(cols, dim=1)

            aux_bool_pos_counts = self.aux_bool_targets.sum(dim=0).tolist()
            self.aux_pos_weights = self._compute_pos_weights(aux_bool_pos_counts)

        # auxiliary float
        self.aux_float_targets = None
        if len(self.aux_float_names) > 0:
            cols = []
            for target in self.aux_float_names:
                assert target in df.columns, f"{target} column not found in dataframe"
                assert df[target].dtype in [float, int], f"{target} column is not float or int"
                col = torch.tensor(df[target].to_numpy(), dtype=torch.float32, device=self.device)
                cols.append(col)

            self.aux_float_targets = torch.stack(cols, dim=1)

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
            if pos_count == 0:
                print("Warning: Positive count is zero, setting weight to 0.0")
                weight = 0.0
            else:
                weight = (total_samples - pos_count) / pos_count
            pos_weights.append(weight)
        
        return pos_weights
    
    def to(self, device: torch.device | str) -> MLFlowDataset:
        """
        Move the dataset to the specified device.
        :param device: Device to move the dataset to
        """
        self.spectra = self.spectra.to(device)
        self.fg_targets = self.fg_targets.to(device)
        if self.aux_bool_targets is not None:
            self.aux_bool_targets = self.aux_bool_targets.to(device)
        if self.aux_float_targets is not None:
            self.aux_float_targets = self.aux_float_targets.to(device)

        self.device = device

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

        out["spectrum"] = spectrum # this is pre-interpolated
        out["fg_targets"] = self.fg_targets[index]

        if len(self.aux_bool_names) > 0:
            assert self.aux_bool_targets is not None, "aux_bool_targets is None, but aux_bool_names is not empty"
            out["aux_bool_targets"] = self.aux_bool_targets[index] 

        if len(self.aux_float_names) > 0:
            assert self.aux_float_targets is not None, "aux_float_targets is None, but aux_float_names is not empty"
            out["aux_float_targets"] = self.aux_float_targets[index] 

        return out


class MLFlowDatasetAggregator:
    """
    - pos weights come from specified list of datasets.
    """

    datasets: dict[str, MLFlowDataset] # e.g. nist:..., chemmotion:..., graphformer:...
    cfg: DictConfig
    dataset_id: str
    split: str
    mask_rate: float

    fg_names: list[str] # names of functional groups

    def __init__(self, cfg: DictConfig, dataset_id: str, split: str, spectrum_transform: Transform | None):
        self.cfg = cfg
        self.dataset_id = dataset_id
        self.split = split

        self.spectrum_transform = spectrum_transform

        self.datasets = {}

        self.download()

    def download(self):

        # master df
        df_path = download_artifact(self.cfg, self.dataset_id, f"{self.split}_df.csv.gz")

        # cache unzipped
        pkl_path = df_path.replace('.csv.gz', '.pkl')
        if os.path.exists(pkl_path):
            print("Loading cached dataset pkl")
            self.df = pd.read_pickle(pkl_path)
        else:
            print("Unzipping dataset...")
            self.df = pd.read_csv(df_path, compression='gzip')
            print("Converting spectra...")
            str_to_arr = lambda st: np.array(ast.literal_eval(st), dtype=np.float32)
            self.df["spectrum"] = self.df["spectrum"].apply(str_to_arr) # type: ignore
            self.df.to_pickle(pkl_path) 

        # split up
        assert "source" in self.df.columns, "source column not found in dataframe"
        if "lser" not in self.df.columns:
            print("Warning: lser column not found in dataframe, assuming all data is non-LSER")
            self.df["lser"] = False

        sources = self.df["source"].unique()

        for source_name in sources:
            for is_lser in [False, True]:
                print(f"Loading dataset for source {source_name} and lser {is_lser}")
                # filter
                df_filtered = self.df[(self.df["source"] == source_name) & (self.df["lser"] == is_lser)]
                if df_filtered.empty:
                    continue

                # create dataset
                dataset = MLFlowDataset(
                    device=self.cfg.device,
                    fg_names=self.cfg.fg_names,
                    aux_bool_names=self.cfg.aux_bool_names,
                    aux_float_names=self.cfg.aux_float_names,
                    spectrum_transform=self.spectrum_transform
                )
                dataset.load_df(df_filtered)
                dataset_name = f"{source_name}_lser" if is_lser else source_name
                self.datasets[dataset_name] = dataset
        
    def get_loader(self, batch_size: int, generate_masks: bool) -> DataLoader:
        dataset_names = list(self.datasets.keys())
        combined_dataset = ConcatDataset([self.datasets[name] for name in dataset_names])
        
        # Calculate the total dataset size (this will be our consistent size)
        total_dataset_size = len(combined_dataset)
        
        # Use custom collate function for mask generation
        collate_fn = self._collate_with_masks_gpu if generate_masks else None
        
        if self.split == "train":
            if len(dataset_names) > 1:
                # Weighted sampling for multi-source training
                weights = []
                for name in dataset_names:
                    dataset = self.datasets[name]
                    weight = getattr(self.cfg, f"{name}_weight")
                    weights += [weight] * len(dataset)
                
                # Create sampler using ALL samples - this maintains consistent epoch size
                sampler = WeightedRandomSampler(
                    weights=weights,
                    num_samples=total_dataset_size,  # Always use full dataset size
                    replacement=True  # Must be True to allow oversampling
                )
                
                dataloader = DataLoader(
                    combined_dataset, 
                    batch_size=batch_size, 
                    sampler=sampler,
                    collate_fn=collate_fn
                )
            else:
                # Regular shuffling for single-source training
                dataloader = DataLoader(
                    combined_dataset, 
                    batch_size=batch_size, 
                    shuffle=True,
                    collate_fn=collate_fn
                )
        else:
            # No masking for validation
            dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)

        return dataloader
        

    def _collate_with_masks_gpu(self, batch):
        """GPU-optimized mask generation"""
        # Standard collation
        collated_batch = torch.utils.data.default_collate(batch)
        
        # Get targets and move to GPU if not already there
        targets = collated_batch["fg_targets"]
        device = targets.device
        
        batch_size, num_targets = targets.shape
        
        # Create mask tensor on GPU
        mask = torch.ones_like(targets, device=device)
        
        # Get mask rates
        min_mask_rate = torch.tensor(self.cfg.min_mask_rate, device=device)
        max_mask_rate = torch.tensor(self.cfg.max_mask_rate, device=device)
        
        # Random mask rate (on GPU)
        global_mask_rate = min_mask_rate + torch.rand(1, device=device) * (max_mask_rate - min_mask_rate)
        min_masks_required = int((batch_size * num_targets * global_mask_rate).item())
        
        # Create one large mask tensor - more vectorized
        flat_targets = targets.view(-1)
        flat_mask = mask.view(-1)
        
        # Get positions of positive and negative examples
        pos_positions = torch.nonzero(flat_targets, as_tuple=True)[0]
        neg_positions = torch.nonzero(flat_targets == 0, as_tuple=True)[0]
        
        # Compute number to mask based on ratios
        pos_ratio = pos_positions.numel() / flat_targets.numel()
        num_pos_to_mask = min(
            pos_positions.numel(),
            int(min_masks_required * min(1.0, pos_ratio * 3))
        )
        num_neg_to_mask = min_masks_required - num_pos_to_mask
        
        # Sample random indices efficiently
        if pos_positions.numel() > 0:
            pos_to_mask = pos_positions[torch.randperm(pos_positions.numel(), device=device)[:num_pos_to_mask]]
            flat_mask[pos_to_mask] = 0
            
        if neg_positions.numel() > 0 and num_neg_to_mask > 0:
            neg_to_mask = neg_positions[torch.randperm(neg_positions.numel(), device=device)[:num_neg_to_mask]]
            flat_mask[neg_to_mask] = 0
        
        # Reshape back and add to batch
        collated_batch["fg_target_mask"] = flat_mask.view(batch_size, num_targets)
        
        return collated_batch