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
                 pred_residual=False,
                 remove_pole=False,          # Whether to remove pole points
                 cell_area=None,
                 engine="netcdf4",
                 device="cpu"
                ):             
        self.device = device 

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
    

class AdjointControlDatasetFromNetCDF:
    """
    Returns adjoint controls at t-1 given adjoint states at t
      x[b] has shape [C_in, H, W] at time  [t]
      y[b] has shape [C_out, H, W] at time [t-1]
    """
    def __init__(self, 
                 state_path, var_name_in, C_in,
                 control_path, var_name_out, C_out,
                 idx_in, idx_out,                 # lists (must be consecutive pairs, e.g., [3,4,5,6,7,8] and [4,5,6,7,8,9])
                 val_percent=0.2,              # percentage of data to use for validation
                 remove_pole=False,          # Whether to remove pole points
                 cell_area=None,
                 engine="netcdf4",
                 device="cpu"
                ):             
        self.device = device 

        # Load the NetCDF file for adjoint states
        ds_state = xr.open_dataset(state_path, engine=engine)
        if remove_pole:
            ref = ds_state[var_name_in].isel(lat=-1, lon=0)  # Reference point at the pole
            ds_state[var_name_in] = ds_state[var_name_in] - ref
        data_state = ds_state[var_name_in].values            # Shape: (N_targets, T, C_in, H, W)
        data_state = torch.tensor(data_state, dtype=torch.float32, device=device)
        ds_state.close()

        # Load the NetCDF file for adjoint controls
        ds_control = xr.open_dataset(control_path, engine=engine)
        data_control = ds_control[var_name_out].values            # Shape: (N_targets, T, C_out, H, W)
        data_control = torch.tensor(data_control, dtype=torch.float32, device=device)
        ds_control.close()

        if cell_area is not None:
            data_state = data_state * cell_area.to(device)
            data_control = data_control * cell_area.to(device)

        N, T, _, H, W = data_control.shape

        assert (len(idx_in) == len(idx_out)), "Expect idx_in and idx_out to have the same length."

        idx_in = np.array(idx_in, dtype=int)
        idx_out = np.array(idx_out, dtype=int)

        x = data_state[:, idx_in, :, :, :]  # (N, T, C_in, H, W)
        y = data_control[:, idx_out, :, :, :]  # (N, T, C_out, H, W)
        import math
        val_count = max(1, math.ceil(val_percent * x.shape[1]))
        split_k = x.shape[1] - val_count
        x_train = x[:, :split_k, :, :, :]
        y_train = y[:, :split_k, :, :, :]
        x_val = x[:, split_k:, :, :, :]
        y_val = y[:, split_k:, :, :, :]

        x_train = x_train.view(-1, C_in, H, W)  # (N * T_train, C_in, H, W)
        y_train = y_train.view(-1, C_out, H, W) # (N * T_train, C_out, H, W)
        x_val   = x_val.view(-1, C_in, H, W)    # (N * T_test, C_in, H, W)
        y_val   = y_val.view(-1, C_out, H, W)   # (N * T_test, C_out, H, W)

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