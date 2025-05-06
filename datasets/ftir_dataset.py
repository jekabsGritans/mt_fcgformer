import os
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from datasets.base_dataset import BaseDataset, Transform


class FTIRDataset(BaseDataset):
    """
    This is the dataset used in the FCG-former paper.
    It contains FTIR spectra and their corresponding functional groups.
    """

    nist_ids: list[int] # (num_samples,) contains the NIST IDs of the samples. not tensor because never used for prediction

    def __init__(self, data_dir: str, split: str, transform: Transform | None, class_names: list[str]):
            """
            Initialize the dataset.
            :param data_dir: Directory containing the dataset
            :param split: Split of the dataset to use. Can be "train", "valid", or "test".
            :param transform: Tensor->Tensor Transform to apply to the input data
            :param class_names: List
            """
            assert split in ["train", "valid", "test"], f"Unknown split: {split}"
            super().__init__(transform, class_names)

            self.data_dir = data_dir
            self.split = split

            self._load_data()

    def _load_data(self):
        # Match all .npy files and extract the ID (without .npy)
        npy_paths = glob(os.path.join(self.data_dir, self.split, "*.npy"))
        ids = [int(os.path.splitext(os.path.basename(path))[0]) for path in npy_paths]
        ids.sort()
        self.nist_ids = ids

        inputs = []
        targets = []
        for nist_id in tqdm(self.nist_ids, desc="Loading data", unit="sample"):
            x, y = self._load_sample(nist_id)
            inputs.append(x)
            targets.append(y)

        self.inputs = torch.stack(inputs, dim=0)  # (num_samples, num_features)
        self.target = torch.stack(targets, dim=0)  # (num_samples, num_classes)

    def _load_sample(self, sample_id):
        """ Load a single sample from the dataset. """
        npy_path = os.path.join(self.data_dir, self.split, f"{sample_id}.npy")
        txt_path = os.path.join(self.data_dir, self.split, f"{sample_id}.txt")

        x = np.load(npy_path)
        with open(txt_path, "r") as f:
            y = np.array([int(tok) for tok in f.read().strip().split()], dtype=np.int64)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        return x, y
    
    def __getitem__(self, index) -> dict:
        """
        Get a sample from the dataset.
        :param index: Index of the sample
        :return: output dictionary
        """

        out = super().__getitem__(index)
        out["nist_id"] = self.nist_ids[index]

        return out