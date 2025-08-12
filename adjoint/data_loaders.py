import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, DistributedSampler


def get_distributed_loaders(train_ds, test_ds, batch_size, num_workers=4, generator=None):
    """
    Wraps datasets with DistributedSampler for multi-node training.
    """
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    test_sampler = DistributedSampler(test_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, generator=generator
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, sampler=test_sampler,
        num_workers=num_workers, pin_memory=True, generator=generator
    )

    return train_loader, test_loader, train_sampler, test_sampler


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
                 pred_residual=False,        # Whether to predict residuals
                 device="cpu",               # Optional torch device
                 engine="netcdf4"            # Engine to use for reading NetCDF
                ):
        self.device = device

        # Load the NetCDF file
        ds = xr.open_dataset(data_path, engine=engine)
        data = ds[var_name].values            # Shape: (N_targets, T, C, H, W)
        data = torch.tensor(data, dtype=torch.float32)

        N, T, C_out, H, W = data.shape
        x_train = data[:, idx_in_train, :C_in]                      # (N, T_train, C_in, H, W)
        y_train = data[:, idx_out_train]                            # (N, T_train, C_out, H, W)
        x_test  = data[:, idx_in_test, :C_in]                       # (N, T_test, C_in, H, W)
        y_test  = data[:, idx_out_test]                             # (N, T_test, C_out, H, W)

        if pred_residual:
            print(f"pred_residual=True → C_in={C_in}, C_out={C_out}")
            y_train[:, :, :C_in] = y_train[:, :, :C_in] - x_train                # Predict residuals
            y_test[:, :, :C_in] = y_test[:, :, :C_in] - x_test

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



class AdjointRolloutDatasetFromNetCDF:
    """
    Returns short sequences for rollout training:
      x_seq[b] has shape [L, C_in, H, W] with times [t, t-1, ..., t-(L-1)]
      y_seq[b] has shape [L-1, C_out, H, W] with times [t-1, ..., t-(L-1)]
    """
    def __init__(self, 
                 data_path, var_name, C_in,
                 idx_in, idx_out,                 # lists (must be consecutive pairs, e.g., [3,4,5,6,7] and [4,5,6,7,8])
                 n_unroll,                        # rollout length during training
                 val_percent=0.2,              # percentage of data to use for validation
                 pred_residual=False,
                 device="cpu", engine="netcdf4"):
        self.device = device
        self.n_unroll = n_unroll
        L = n_unroll + 1

        ds = xr.open_dataset(data_path, engine=engine)
        data = torch.tensor(ds[var_name].values, dtype=torch.float32)   # (N, T, C, H, W)
        ds.close()

        N, T, C_out, H, W = data.shape
        x = data[:, idx_in, :C_in]      # (N, T_in, C_in, H, W)
        y = data[:, idx_out]            # (N, T_out, C_out, H, W)

        if pred_residual:
            print(f"pred_residual=True → C_in={C_in}, C_out={C_out}")
            y = y.clone()
            y[:, :, :C_in] = y[:, :, :C_in] - x

        # We will build windows along the time dimension of x/y:
        # window k uses x[:, k : k+L] -> inputs, and y[:, k : k+n_unroll] -> targets
        T_in = x.shape[1]
        assert T_in >= L, "idx_in too short for requested n_unroll"
        # also need the aligned y to have at least n_unroll targets after k
        T_out = y.shape[1]
        assert T_out >= n_unroll, "idx_out too short for requested n_unroll"
        assert T_in == T_out + 1 or T_in == T_out, "Expect x times and y times to be aligned consecutively (x[t] -> y[t-1])."

        num_windows = T_in - L + 1
        x_win_list, y_win_list = [], []
        for k in range(num_windows):
            x_win_list.append(x[:, k:k+L])             # (N, L, C_in, H, W)
            y_win_list.append(y[:, k:k+n_unroll])      # (N, n_unroll, C_out, H, W)

        # Chronological split: last >=20% windows are validation
        import math
        val_count = max(1, math.ceil(val_percent * num_windows))
        split = num_windows - val_count   # train windows: [0..split-1], val windows: [split..end]
        assert split > 0, "val_percent too high for num_windows"

        # Stack windows across batch (N * num_windows, ...)
        x_train = torch.cat(x_win_list[:split], dim=0) if split > 0 else None   # (N*num_windows*train_percent, L, C_in, H, W)
        y_train = torch.cat(y_win_list[:split], dim=0) if split > 0 else None   # (N*num_windows*train_percent, n_unroll, C_out, H, W)
        x_val   = torch.cat(x_win_list[split:], dim=0)                          # (N*num_windows*val_percent, L, C_in, H, W)
        y_val   = torch.cat(y_win_list[split:], dim=0)                          # (N*num_windows*val_percent, n_unroll, C_out, H, W)

        self.train = (x_train, y_train) if split > 0 else None
        self.val   = (x_val, y_val)

    def get_datasets(self):
        """
        Returns: (train_ds, val_ds)
        If there are too few windows for a train split, train_ds will be None.
        """
        train_ds = None
        if self.train is not None:
            x_tr, y_tr = self.train
            train_ds = TensorDataset(x_tr, y_tr)
        x_va, y_va = self.val
        val_ds = TensorDataset(x_va, y_va)
        return train_ds, val_ds


        
