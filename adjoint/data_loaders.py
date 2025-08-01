import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

class AdjointDatasetFromNetCDF:
    def __init__(self, 
                 data_path,                  # Path to NetCDF file
                 var_name,                   # Variable name in the file for data
                 C_in,                       # Number of input channels
                 idx_in_train,               # Temporal index of input variable in training set
                 idx_out_train,              # Temporal index of output variable in training set
                 idx_in_test,                # Temporal index of input variable in test set
                 idx_out_test,               # Temporal index of output variable in test set
                 label=None,                 # Optional variable name for labels
                 device="cpu",               # Optional torch device
                 engine="netcdf4"            # Engine to use for reading NetCDF
                ):
        self.device = device

        # Load the NetCDF file
        ds = xr.open_dataset(data_path, engine=engine)
        data = ds[var_name].values            # Shape: (N_targets, T, C, H, W)
        data = torch.tensor(data, dtype=torch.float32).to(device)

        N, T, C_out, H, W = data.shape
        x_train = data[:, idx_in_train, :C_in]                      # (N, T_train, C_in, H, W)
        y_train = data[:, idx_out_train]                            # (N, T_train, C_out, H, W)
        x_test  = data[:, idx_in_test, :C_in]                       # (N, T_test, C_in, H, W)
        y_test  = data[:, idx_out_test]                             # (N, T_test, C_out, H, W)


        # Flatten spatial targets and time: (N*T, C, H, W)
        self.x_train = x_train.reshape(-1, C_in, H, W)
        self.y_train = y_train.reshape(-1, C_out, H, W)
        self.x_test  = x_test.reshape(-1, C_in, H, W)
        self.y_test  = y_test.reshape(-1, C_out, H, W)

        # Compute L2 norms over channel, height, width: shape → (N*T,)
        # self.x_train_norms = torch.norm(self.x_train.view(self.x_train.shape[0], -1), dim=1)
        # self.x_test_norms = torch.norm(self.x_test.view(self.x_test.shape[0], -1), dim=1)

        # Compute maximum of the absolute value over channel, height, width: shape → (N*T,)
        self.x_train_norms = self.x_train.abs().amax(dim=(1,2,3))
        self.x_test_norms = self.x_test.abs().amax(dim=(1,2,3))

        # Normalize input and output by input norm
        self.x_train = self.x_train / self.x_train_norms.view(-1, 1, 1, 1)
        self.y_train = self.y_train / self.x_train_norms.view(-1, 1, 1, 1)
        self.x_test = self.x_test / self.x_test_norms.view(-1, 1, 1, 1)
        self.y_test = self.y_test / self.x_test_norms.view(-1, 1, 1, 1)

        self.train_labels = None
        self.test_labels = None

        if label is not None:
            labels = torch.tensor(label, dtype=torch.float32).to(device)  # (N, C_in)

            # Expand to (N, T, C_in) → then flatten to (N*T, C_in)
            n_train  = len(idx_in_train)
            n_test = len(idx_in_test)
            train_labels = labels[:, None, :].expand(-1, n_train, -1)                   # (N, T_train, C_in)
            test_labels = labels[:, None, :].expand(-1, n_test, -1)                   # (N, T_test, C_in)
            self.train_labels = train_labels.reshape(-1, train_labels.shape[-1])                   # (N*T_train, C_in)
            self.test_labels = test_labels.reshape(-1, test_labels.shape[-1])                   # (N*T_test, C_in)

        ds.close()

    def get_datasets(self):
        if self.train_labels is None:
            train_ds = TensorDataset(self.x_train, self.y_train)
            test_ds  = TensorDataset(self.x_test, self.y_test)
        else:
            train_ds = TensorDataset(self.x_train, self.y_train, self.train_labels)
            test_ds  = TensorDataset(self.x_test, self.y_test, self.test_labels)

        return train_ds, test_ds
    
    def get_norms(self):
        return self.x_train_norms, self.x_test_norms
    

class WetMaskFromNetCDF:
    def __init__(self, 
                 wet_path,                  # Path to NetCDF file
                 var_name,                   # Variable name in the file for data
                 device="cpu",               # Optional torch device
                 engine="netcdf4"            # Engine to use for reading NetCDF
                ):
        self.device = device

        ds = xr.open_dataset(wet_path, engine=engine)
        data = ds[var_name].values            # Shape: (H, W)
        self.wet_mask = torch.tensor(data, dtype=torch.float32).to(device)
        ds.close()

    def get_wet_mask(self):
        return self.wet_mask
        
