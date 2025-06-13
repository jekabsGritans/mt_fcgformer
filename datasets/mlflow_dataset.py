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
        
    def get_loader(self, batch_size: int) -> GPUBatchSampler | DataLoader:
        if self.split == "train":
            # Get dataset weights
            dataset_weights = {}
            for name in self.datasets.keys():
                weight = getattr(self.cfg, f"{name}_weight", 0.0)
                dataset_weights[name] = weight
                
            # Create GPU batch sampler
            gpu_sampler = GPUBatchSampler(
                self.datasets, 
                dataset_weights,
                batch_size, 
                device=self.cfg.device
            )
            
            # Return iterator directly (no DataLoader needed)
            return gpu_sampler
        else:
            # For validation, keep current DataLoader
            combined_dataset = ConcatDataset([self.datasets[name] for name in self.datasets.keys()])
            return DataLoader(
                combined_dataset, 
                batch_size=batch_size,
                shuffle=False,
                pin_memory=False
            )

    

class GPUBatchSampler:
    """Sample batches directly on GPU for maximum throughput"""
    def __init__(self, datasets, weights, batch_size, device, epochs_worth=1):
        self.device = device
        self.batch_size = batch_size
        
        # Calculate total samples and weights
        self.n_samples = sum(len(dataset) for dataset in datasets.values())
        
        # Create mapping from flat index to (dataset_name, dataset_idx)
        self.index_map = []
        self.dataset_tensors = {}
        
        # Pre-calculate weights tensor for all samples
        all_weights = []
        
        for name, dataset in datasets.items():
            # Get weight for this dataset
            weight = weights.get(name, 0.0)
            if weight <= 0.0:
                continue
                
            # Add dataset to mapping
            ds_size = len(dataset)
            self.index_map.extend([(name, i) for i in range(ds_size)])
            
            # Store dataset tensors
            self.dataset_tensors[name] = {
                "spectrum": dataset.spectra,
                "fg_targets": dataset.fg_targets
            }
            
            if dataset.aux_bool_targets is not None:
                self.dataset_tensors[name]["aux_bool_targets"] = dataset.aux_bool_targets
                
            if dataset.aux_float_targets is not None:
                self.dataset_tensors[name]["aux_float_targets"] = dataset.aux_float_targets
            
            # Add weights
            all_weights.extend([weight] * ds_size)
        
        # Convert weights to tensor for GPU sampling
        weights_tensor = torch.tensor(all_weights, device=device)
        self.n_valid_samples = len(weights_tensor)
        
        # Pre-generate multiple epochs worth of batch indices
        n_batches = self.n_valid_samples // batch_size * epochs_worth
        
        # Generate indices using multinomial sampling on GPU
        self.batches = []
        for _ in range(n_batches):
            batch_indices = torch.multinomial(
                weights_tensor, 
                batch_size,
                replacement=True
            )
            self.batches.append(batch_indices)
            
        self.current_batch = 0
        
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.current_batch >= len(self.batches):
            # Regenerate batches
            self.current_batch = 0
            raise StopIteration
            
        indices = self.batches[self.current_batch]
        batch = {}
        
        # Get index mappings for this batch
        batch_mappings = [self.index_map[i] for i in indices.cpu().numpy()]
        
        # Group by dataset for efficient tensor indexing
        dataset_indices = {}
        for ds_name, idx in batch_mappings:
            if ds_name not in dataset_indices:
                dataset_indices[ds_name] = []
            dataset_indices[ds_name].append(idx)
            
        # Create batch directly using tensor indexing on GPU
        for key in ["spectrum", "fg_targets"]:
            tensors_to_stack = []
            
            for ds_name, indices in dataset_indices.items():
                if not indices:
                    continue
                    
                # Convert to tensor
                idx_tensor = torch.tensor(indices, device=self.device)
                ds_tensor = self.dataset_tensors[ds_name][key][idx_tensor]
                tensors_to_stack.append(ds_tensor)
                
            batch[key] = torch.cat(tensors_to_stack, dim=0)
            
        # Similarly for aux targets
        for key in ["aux_bool_targets", "aux_float_targets"]:
            tensors_to_stack = []
            
            for ds_name, indices in dataset_indices.items():
                if not indices or key not in self.dataset_tensors[ds_name]:
                    continue
                    
                idx_tensor = torch.tensor(indices, device=self.device)
                ds_tensor = self.dataset_tensors[ds_name][key][idx_tensor]
                tensors_to_stack.append(ds_tensor)
                
            if tensors_to_stack:
                batch[key] = torch.cat(tensors_to_stack, dim=0)
                
        self.current_batch += 1
        return batch
        
    def __len__(self):
        return len(self.batches)