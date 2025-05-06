import os
from glob import glob

import numpy as np
from tqdm import tqdm

from datasets.base_dataset import BaseDataset, Transform
from utils.transforms import np_to_torch


class FTIRDataset(BaseDataset):
    """
    This is the dataset used in the FCG-former paper.
    It contains FTIR spectra and their corresponding functional groups.
    """

    nist_ids: np.ndarray # (num_samples,) contains the NIST IDs of the samples

    def __init__(self, data_dir: str, split: str, class_names: list[str]):
            """
            Initialize the dataset.
            :param data_dir: Directory containing the dataset
            :param split: Split of the dataset to use. Can be "train", "valid", or "test".
            :param class_names: Names of boolean features that are predicted
            """
            assert split in ["train", "valid", "test"], f"Unknown split: {split}"
            self.data_dir = data_dir
            self.split = split
            self.class_names = class_names

            # This is fixed per dataset. I.e. we won't want to change with params. 
            # For training augmentations prolly fixed as well, but stochastic and myb dependant on state.
            self.transform = np_to_torch

            super().__init__()
    
    def load_data(self):
        # Match all .npy files and extract the ID (without .npy)
        npy_paths = glob(os.path.join(self.data_dir, self.split, "*.npy"))
        ids = [int(os.path.splitext(os.path.basename(path))[0]) for path in npy_paths]
        ids.sort()
        self.nist_ids = np.array(ids, dtype=np.int64)

        inputs = []
        targets = []
        for nist_id in tqdm(self.nist_ids, desc="Loading data", unit="sample"):
            x, y = self._load_sample(nist_id)
            inputs.append(x)
            targets.append(y)
        
        self.inputs = np.stack(inputs, axis=0)
        self.target = np.stack(targets, axis=0)
    
    def __getitem__(self, index) -> dict:
        """
        Get a sample from the dataset.
        :param index: Index of the sample
        :return: output dictionary
        """

        out = super().__getitem__(index)
        out["nist_id"] = self.nist_ids[index]

        return out

    def _load_sample(self, sample_id):
        """ Load a single sample from the dataset. """
        npy_path = os.path.join(self.data_dir, self.split, f"{sample_id}.npy")
        txt_path = os.path.join(self.data_dir, self.split, f"{sample_id}.txt")

        x = np.load(npy_path)
        with open(txt_path, "r") as f:
            y = np.array([int(tok) for tok in f.read().strip().split()], dtype=np.int64)
        return x, y