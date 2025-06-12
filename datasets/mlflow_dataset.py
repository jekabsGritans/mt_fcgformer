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

        self.mask_rate = cfg.mask_rate if split == "train" else 0.0

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

        # Use custom collate function for mask generation
        collate_fn = self._collate_with_masks if generate_masks else None
        
        if self.split == "train":
            if len(dataset_names) > 1:
                # Weighted sampling for multi-source training
                weights = []
                for name in dataset_names:
                    dataset = self.datasets[name]
                    weight = getattr(self.cfg, f"{name}_weight")
                    weights += [weight] * len(dataset)

                num_nonzero_weights = torch.tensor(weights).nonzero().numel()
                sampler = WeightedRandomSampler(weights, num_samples=num_nonzero_weights, replacement=True)
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
    
    def _collate_with_masks(self, batch):
        """Custom collate function that adds random masks to batches"""
        # Standard collation
        collated_batch = torch.utils.data.default_collate(batch)
        
        # Add masks to batch
        targets = collated_batch["fg_targets"]
        batch_size, num_targets = targets.shape
        total_labels = batch_size * num_targets
        
        # Create mask tensor (1 = use this label, 0 = mask/unknown)
        mask = torch.ones_like(targets)
        
        # Choose a random global mask percentage between 50% and 100%
        global_mask_rate = torch.rand(1).item() * 0.5 + 0.5  # Random between 0.5 and 1.0
        min_masks_required = int(total_labels * global_mask_rate)
        
        # Apply class-aware masking strategy
        total_masked = 0
        
        # First pass - apply class-aware masking
        for i in range(num_targets):
            # Get class distribution for this target
            positives = targets[:, i].float().mean().item()
            
            # Higher base mask rate to get closer to our 50-100% target
            base_mask_rate = 0.7
            
            # Adjust mask rate based on class imbalance
            # Rare positive classes get masked slightly less often
            pos_mask_rate = base_mask_rate * min(1.0, positives * 5)
            neg_mask_rate = base_mask_rate * 1.2  # Mask negatives more aggressively
            
            # Create masks for positive and negative examples separately
            pos_indices = targets[:, i].nonzero().squeeze(-1)
            neg_indices = (targets[:, i] == 0).nonzero().squeeze(-1)
            
            # Randomly mask positive examples
            if len(pos_indices) > 0:
                num_pos_to_mask = max(1, int(len(pos_indices) * pos_mask_rate))
                pos_to_mask = pos_indices[torch.randperm(len(pos_indices))[:num_pos_to_mask]]
                mask[pos_to_mask, i] = 0
                total_masked += len(pos_to_mask)
            
            # Randomly mask negative examples
            if len(neg_indices) > 0:
                num_neg_to_mask = max(1, int(len(neg_indices) * neg_mask_rate))
                neg_to_mask = neg_indices[torch.randperm(len(neg_indices))[:num_neg_to_mask]]
                mask[neg_to_mask, i] = 0
                total_masked += len(neg_to_mask)
        
        # If we haven't masked enough labels, mask more randomly
        if total_masked < min_masks_required:
            # Create a mask of currently unmasked positions
            additional_needed = min_masks_required - total_masked
            unmasked = mask.nonzero()
            
            # Randomly select additional positions to mask
            if len(unmasked) > 0:
                indices = torch.randperm(len(unmasked))[:min(additional_needed, len(unmasked))]
                additional_to_mask = unmasked[indices]
                mask[additional_to_mask[:, 0], additional_to_mask[:, 1]] = 0
        
        # Add mask to batch
        collated_batch["fg_target_mask"] = mask
        
        return collated_batch