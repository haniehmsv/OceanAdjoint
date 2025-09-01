import torch.distributed as dist
import xarray as xr
import numpy as np
import torch
import netCDF4 as nc
from torch.utils.data import Dataset, DataLoader, TensorDataset, DistributedSampler


def get_distributed_loaders(train_ds, test_ds, batch_size, num_workers=4, generator=None, pin_memory=False):
    """
    Wraps datasets with DistributedSampler for multi-node training.
    """
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory, generator=generator
    )

    if test_ds is None:
        test_loader = None
        test_sampler = None
    else:
        test_sampler = DistributedSampler(test_ds, shuffle=False)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, sampler=test_sampler,
            num_workers=0, pin_memory=pin_memory, generator=generator
        )

    return train_loader, test_loader, train_sampler, test_sampler
    

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
      x_seq[b] has shape [n_unroll, C_in, H, W] with times [t, t-1, ..., t-n_unroll+1]
      y_seq[b] has shape [n_unroll, C_out, H, W] with times [t-1, ..., t-n_unroll]
    """
    def __init__(self, 
                 data_path, var_name, C_in,
                 idx_in, idx_out,                 # lists (must be consecutive pairs, e.g., [3,4,5,6,7,8] and [4,5,6,7,8,9])
                 n_unroll,                        # rollout length during training
                 val_percent=0.2,              # percentage of data to use for validation
                 wet = None,
                 pred_residual=False,
                 remove_pole=False,          # Whether to remove pole points
                 cell_area=None,
                 engine="netcdf4",
                 device="cpu"
                ):             
        self.device = device 
        wet = wet.to(device)

        # Load the NetCDF file
        ds = xr.open_dataset(data_path, engine=engine)
        data = ds[var_name].values            # Shape: (N_targets, T, C, H, W)
        data = torch.tensor(data, dtype=torch.float32, device=device)
        if remove_pole:
            wet_mask = (wet > 0).to(data.dtype)  
            ref = data[:, :, :, -1, 0]  # Reference point at the pole
            data = data - ref[..., None, None] * wet_mask[None, None, None, :, :]
            wet[-1, 0] = 0  # Remove the pole point
        ds.close()
        if cell_area is not None:
            data = data * cell_area.to(device)

        N, T, C_out, H, W = data.shape

        assert len(idx_in) >= n_unroll, "idx_out too short for requested n_unroll"
        assert (len(idx_in) == len(idx_out) + 1) or (len(idx_in) == len(idx_out)), "Expect idx_in and idx_out to be aligned consecutively (x[t] -> y[t-1])."

        idx_in = np.array(idx_in, dtype=int)
        idx_out = np.array(idx_out, dtype=int)

        # Build (n_case, t_start) pairs for all windows, then split chronologically
        num_windows = len(idx_in) - n_unroll + 1

        # chronological split by k
        import math
        val_count = max(1, math.ceil(val_percent * num_windows))
        split_k = num_windows - val_count   # windows [0..split_k-1] for train; [split_k..] val
        if split_k <= 0:
            raise RuntimeError(
                f"val_percent={val_percent} is too high for num_windows={num_windows} "
                f"(train windows would be {split_k} ≤ 0). Reduce val_percent or increase data length."
            )

        data_mean = data[:,idx_in[0]:(idx_in[-val_count]+1)].mean(dim=(0,1)) # [C, H, W]
        data_std = data[:,idx_in[0]:(idx_in[-val_count]+1)].std(dim=(0,1)) # [C, H, W]
        wet_bool = (wet > 0)
        data_mean[:, ~wet_bool] = 0
        data_std[:,  ~wet_bool] = 1
        zero_std = data_std.abs() == 0.0
        data_std = data_std.masked_fill(zero_std, 1.0)
        self.data_mean = data_mean
        self.data_std = data_std
        data = (data - self.data_mean) / self.data_std  # Normalize the data

        x_window = []
        y_window = []
        for k in range(num_windows):
            t_in = idx_in[k : k + n_unroll]                # length n_unroll
            t_out = idx_out[k : k + n_unroll]       # length n_unroll
            x = data[:, t_in, :C_in, :, :]          # (N, n_unroll, C_in, H, W)
            y = data[:, t_out, :, :, :]                  # (N, n_unroll, C_out, H, W)
            x_window.append(x)
            y_window.append(y)
        
        # Stack windows across batch (N * num_windows, ...)
        x_train = torch.cat(x_window[:split_k], dim=0)                          # (N*num_windows*train_percent, n_unroll, C_in, H, W)
        y_train = torch.cat(y_window[:split_k], dim=0)                          # (N*num_windows*train_percent, n_unroll, C_out, H, W)
        x_val   = torch.cat(x_window[split_k:], dim=0)                          # (N*num_windows*val_percent, n_unroll, C_in, H, W)
        y_val   = torch.cat(y_window[split_k:], dim=0)                          # (N*num_windows*val_percent, n_unroll, C_out, H, W)

        if pred_residual:
            # y[..., :C_in] := y - x_prev  (work in CPU, no in-place on saved tensors)
            # x_prev is x at t_in[1:] aligned with y at t_out[:]
            y_train -= x_train
            y_val -= x_val


        self.train = (x_train, y_train)
        self.val   = (x_val, y_val)

    def get_datasets(self):
        """
        Returns: (train_ds, val_ds)
        """
        x_tr, y_tr = self.train
        train_ds = TensorDataset(x_tr, y_tr)
        x_va, y_va = self.val
        val_ds = TensorDataset(x_va, y_va)
        return train_ds, val_ds
    
    def get_mean_std(self):
        """
        Returns the mean and std of training dataused for normalization.
        """
        return self.data_mean, self.data_std
    

class AdjointForcingDatasetFromNetCDF:
    """
    Returns short sequences for training:
      x_seq[b] has shape [n_unroll, C_in, H, W] with times [t, t-1, ..., t-n_unroll+1]
      y_seq[b] has shape [n_unroll, C_out_total, H, W] with times [t-1, ..., t-n_unroll]
    """
    def __init__(self, 
                 path_in, var_name_in, C_in,
                 path_out, var_name_out, C_out_total,
                 idx_in, idx_out,                 # lists (must be consecutive pairs, e.g., [3,4,5,6,7,8] and [4,5,6,7,8,9])
                 n_unroll,                        # rollout length during training
                 val_percent=0.2,              # percentage of data to use for validation
                 wet = None,
                 pred_residual=False,
                 remove_pole=False,          # Whether to remove pole points
                 engine="netcdf4",
                 device="cpu"
                ):             
        self.device = device 
        wet = wet.to(device)

        # Load the NetCDF file for input data
        ds = xr.open_dataset(path_in, engine=engine)
        data_in = ds[var_name_in].values 
        ds.close()           
        data_in = torch.tensor(data_in, dtype=torch.float32, device=device)     # Shape: (N_targets, T_in, C_in, H, W)
        if remove_pole:
            wet_mask = (wet > 0).to(data_in.dtype)  
            ref = data_in[:, :, :, -1, 0]  # Reference point at the pole
            data_in = data_in - ref[..., None, None] * wet_mask[None, None, None, :, :]
            wet[-1, 0] = 0  # Remove the pole point

        # Load the NetCDF file for output data
        ds = xr.open_dataset(path_out, engine=engine)
        data_out = ds[var_name_out].values      
        ds.close()      
        data_out = torch.tensor(data_out, dtype=torch.float32, device=device)     # Shape: (N_targets, T_out, C_out, H, W)

        N, T_in, C_in, H, W = data_in.shape
        _, T_out, C_out, _, _ = data_out.shape

        idx_in = np.array(idx_in, dtype=int)
        idx_out = np.array(idx_out, dtype=int)
        data_combined = torch.cat([data_in, data_out], dim=2)  # Concatenate along channel dimension (C_in + C_out)
        num_windows = len(idx_in) - n_unroll + 1

        # chronological split by k
        import math
        val_count = max(1, math.ceil(val_percent * num_windows))
        split_k = num_windows - val_count   # windows [0..split_k-1] for train; [split_k..] val
        if split_k <= 0:
            raise RuntimeError(
                f"val_percent={val_percent} is too high for num_windows={num_windows} "
                f"(train windows would be {split_k} ≤ 0). Reduce val_percent or increase data length."
            )

        data_mean = data_combined[:,idx_in[0]:(idx_in[-val_count]+1)].mean(dim=(0,1)) # [C_in+C_out, H, W]
        data_std = data_combined[:,idx_in[0]:(idx_in[-val_count]+1)].std(dim=(0,1)) # [C_in+C_out, H, W]
        del data_combined
        wet_bool = (wet > 0)
        data_mean[:, ~wet_bool] = 0
        data_std[:,  ~wet_bool] = 1
        zero_std = data_std.abs() == 0.0
        data_std = data_std.masked_fill(zero_std, 1.0)
        self.data_mean = data_mean
        self.data_std = data_std
        data_in = (data_in - self.data_mean[:C_in]) / self.data_std[:C_in]  # Normalize the input data
        data_out = (data_out - self.data_mean[C_in:]) / self.data_std[C_in:]  # Normalize the output data

        if n_unroll > 1:
            print(f"n_unroll > 1 (={n_unroll}). The output will be both ad-states and ad-forcings.")
            C_out_total = C_in + C_out  # Predict both states and forcings
        
        x_slice = data_in[:, idx_in[0] : idx_in[-1]  + 1]     # [N, Tx, Cin, H, W]
        y_slice = data_out[:, idx_out[0] : idx_out[-1] + 1]   # [N, Ty, Cout_tot, H, W]
        # Unfold along time (size=n_unroll, step=1) -> views
        X = x_slice.unfold(1, n_unroll, 1)  # [N, num_windows, Cin, H, W, n_unroll]
        X = X.movedim(-1, 2)                # [N, num_windows, n_unroll, Cin, H, W]
        Y = y_slice.unfold(1, n_unroll, 1)  # [N, num_windows, C_out, H, W, n_unroll]
        Y = Y.movedim(-1, 2)                # [N, num_windows, n_unroll, C_out, H, W]

        if C_out_total > C_out:
            Y = torch.cat([X, Y], dim=3)  # Concatenate along channel dimension (C_in + C_out)

        X_train, Y_train = X[:, :split_k], Y[:, :split_k]
        X_val,   Y_val   = X[:, split_k:], Y[:, split_k:]

        X_train = X_train.reshape(-1, *X_train.shape[2:])  # [N*K, L, Cin, H, W]
        Y_train = Y_train.reshape(-1, *Y_train.shape[2:])  # [N*K, L, Cout_tot, H, W]
        X_val   = X_val.reshape(-1, *X_val.shape[2:])      # [N*K_val, L, Cin, H, W]
        Y_val   = Y_val.reshape(-1, *Y_val.shape[2:])      # [N*K_val, L, Cout_tot, H, W]

        if pred_residual:
            # y[..., :C_in] := y - x_prev  (work in CPU, no in-place on saved tensors)
            # x_prev is x at t_in[1:] aligned with y at t_out[:]
            Y_train[:, :, :C_in] = Y_train[:, :, :C_in] - X_train
            Y_val[:, :, :C_in] = Y_val[:, :, :C_in] - X_val

        self.train = (X_train, Y_train)
        self.val   = (X_val, Y_val)

    def get_datasets(self):
        """
        Returns: (train_ds, val_ds)
        """
        x_tr, y_tr = self.train
        train_ds = TensorDataset(x_tr, y_tr)
        x_va, y_va = self.val
        val_ds = TensorDataset(x_va, y_va)
        return train_ds, val_ds
    
    def get_mean_std(self):
        """
        Returns the mean and std of training dataused for normalization.
        """
        return self.data_mean, self.data_std