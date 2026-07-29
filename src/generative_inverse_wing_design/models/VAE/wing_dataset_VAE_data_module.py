from __future__ import annotations

import random

import aerosandbox.numpy as np
from omegaconf import DictConfig
import pandas as pd
import pytorch_lightning as L
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from generative_inverse_wing_design.DataGenerator.utils.wing_dataset_utils import (
    map_afs_cst_params,
)


class Wing_Dataset(Dataset):
    def __init__(self, data_path: str, labels_loaded: bool):
        # Read CSV file as DataFrame first
        df = pd.read_csv(data_path)
        self.df = df

        # Check and drop cluster column if it exists

        sampled_parm_cols = [
            "chord_ctrl_pos",
            "chord_weight_ctrl",
            "chord_tip",
            "twist_root",
            "twist_tip",
            "twist_mean_pos",
            "spar_sweep",
        ]

        afs = map_afs_cst_params(self.df["aeroelastic_wing"])

        params = self.df[sampled_parm_cols].values.astype("float32")

        performance_metrics = [
            "L",
            "D",
            "CL",
            "CD",
            "CM",
            "CMa",
            "CLa",
            "max_heave_displacement",
            "max_twist_displacement",
            "max_bending_moment",
            "max_von_mises_stress",
            "mass",
        ]
        #  "CL/CD" (is the last metric, computed below)
        perf_metrics_data = self.df[performance_metrics].values.astype("float32")
        cl_over_cd = perf_metrics_data[:, 2:3] / perf_metrics_data[:, 3:4]

        data_with_metrics = np.concatenate([params, afs, perf_metrics_data, cl_over_cd], axis=1)

        self.data = data_with_metrics
        self.labels_loaded = labels_loaded

    @property
    def scaler(self) -> StandardScaler:
        if not self.labels_loaded:
            # Exclude last 13 performance metrics
            scaler = StandardScaler().fit(self.data[:, :-13])
        else:
            scaler = StandardScaler().fit(self.data)
        return scaler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if not self.labels_loaded:
            sample = sample[:-13]  # Exclude last 13 performance metrics
        return torch.tensor(sample)


class WingDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig, seed: int, load_labels: bool = False):
        super().__init__()
        self.batch_size: int = cfg.batch_size
        self.dataset_path: str = cfg.dataset_path
        self.train_split: float = cfg.train_split
        self.val_split: float = cfg.val_split
        self.train_subset_fraction: float = float(
            cfg.train_subset_fraction if "train_subset_fraction" in cfg else 1.0
        )
        self.labels_loaded: bool = load_labels
        self.seed: int = seed

        L.seed_everything(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        self.dataset = Wing_Dataset(self.dataset_path, self.labels_loaded)

        # Split into train/val/test
        train_size = int(len(self.dataset) * self.train_split)
        val_size = int(len(self.dataset) * self.val_split)
        test_size = len(self.dataset) - train_size - val_size

        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size], generator=generator
        )

        if self.train_subset_fraction < 1.0:
            subset_size = max(1, int(len(self.train_dataset) * self.train_subset_fraction))
            subset_indices = self.train_dataset.indices[:subset_size]
            self.train_dataset = Subset(self.dataset, subset_indices)
            print(
                f"Applied train subset fraction {self.train_subset_fraction:.6f} "
                f"-> using {subset_size} training samples."
            )

        print(
            f"Split sizes - Train: {len(self.train_dataset)}, "
            f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}"
        )

        # allow for a check if splits are the same as for training
        train_indices = self.train_dataset.indices
        test_indices = self.test_dataset.indices

        print(f"First 10 train indices: {train_indices[:10]}")
        print(f"First 10 test indices: {test_indices[:10]}")

    @property
    def scaler(self) -> StandardScaler:
        train_indices = np.asarray(self.train_dataset.indices, dtype=int)
        train_data = self.dataset.data[train_indices]
        if not self.labels_loaded:
            train_data = train_data[:, :-13]
        return StandardScaler().fit(train_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def load_labels(self) -> None:
        """Loads Labels"""
        self.dataset.labels_loaded = True
        self.labels_loaded = True

    def unload_labels(self) -> None:
        """Unloads Labels"""
        self.dataset.labels_loaded = False
        self.labels_loaded = False
