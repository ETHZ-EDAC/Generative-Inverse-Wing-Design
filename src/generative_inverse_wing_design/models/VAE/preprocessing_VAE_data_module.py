import pytorch_lightning as L
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import pandas as pd
from pandas import DataFrame
import torch
from sklearn.preprocessing import StandardScaler
import aerosandbox.numpy as np

class Airfoil_Dataset(Dataset):
    def __init__(self, data_path: str):
        # Read CSV file as DataFrame first
        aero_data = pd.read_csv(data_path)
        
        # Check and drop cluster column if it exists
        if "cluster" in aero_data.columns:
            aero_data = aero_data.drop(columns=['cluster'])
            print("Dropped 'cluster' column for dimensionality reduction.")
        
        # Select the columns we want to use
        feature_columns = ['asb_leading_edge_weight'] + \
                            [f'asb_upper_{i}' for i in range(3)] + \
                            [f'asb_lower_{i}' for i in range(3)] + \
                            ['CL', 'CD', 'CM']

        # Convert to float32 array
        aero_data_values = aero_data[feature_columns].values.astype('float32')
        
        # Standardize the data
        self.scaler = StandardScaler()
        aero_data_scaled = self.scaler.fit_transform(aero_data_values)
        
        self.data = aero_data_scaled

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample)


class AirfoilDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.batch_size: int = cfg.batch_size
        self.dataset_path: str = cfg.dataset_path
        self.train_split: float = cfg.train_split
        self.val_split: float = cfg.val_split

        self.dataset = Airfoil_Dataset(self.dataset_path)
            
        # Split into train/val/test
        train_size = int(len(self.dataset) * self.train_split)
        val_size = int(len(self.dataset) * self.val_split)
        test_size = len(self.dataset) - train_size - val_size
        
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )
        
        print(f"Split sizes - Train: {len(self.train_dataset)}, "
              f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        
    @property
    def scaler(self):
        return self.dataset.scaler

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)