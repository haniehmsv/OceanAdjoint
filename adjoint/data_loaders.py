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
    test_sampler = DistributedSampler(test_ds, shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory, generator=generator
    )
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
        if remove_pole:
            ref = ds[var_name].isel(lat=-1, lon=0)  # Reference point at the pole
            ds[var_name] = ds[var_name] - ref
        data = ds[var_name].values            # Shape: (N_targets, T, C, H, W)
        data = torch.tensor(data, dtype=torch.float32, device=device)
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

        data_mean = data[:,idx_in[0]:idx_in[0]+split_k].mean(dim=(0,1)) # [C, H, W]
        data_std = data[:,idx_in[0]:idx_in[0]+split_k].std(dim=(0,1)) # [C, H, W]
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
    