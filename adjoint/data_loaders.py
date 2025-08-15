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
      x_seq[b] has shape [L, C_in, H, W] with times [t, t-1, ..., t-(L-1)]
      y_seq[b] has shape [L-1, C_out, H, W] with times [t-1, ..., t-(L-1)]
    """
    def __init__(self, 
                 data_path, var_name, C_in,
                 idx_in, idx_out,                 # lists (must be consecutive pairs, e.g., [3,4,5,6,7,8,9] and [4,5,6,7,8,9])
                 n_unroll,                        # rollout length during training
                 val_percent=0.2,              # percentage of data to use for validation
                 pred_residual=False,
                 remove_pole=False,          # Whether to remove pole points
                 cell_area=None,
                 engine="netcdf4"):
        self.data_path = data_path
        self.var_name = var_name
        self.C_in = C_in
        self.pred_residual = pred_residual
        self.remove_pole = remove_pole
        self.cell_area = cell_area.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W) for broadcasting

        # Open once to get shapes, then close (workers will reopen lazily)
        with nc.Dataset(self.data_path, "r") as ds:
            v = ds.variables[self.var_name]  # expected shape (N, T, C_out, H, W)
            self.N, self.T, self.C_out, self.H, self.W = v.shape

        self.n_unroll = n_unroll
        self.L = self.n_unroll + 1

        assert len(idx_in) >= self.L and len(idx_out) >= self.n_unroll, "idx_in too short for requested n_unroll"
        assert (len(idx_in) == len(idx_out) + 1) or (len(idx_in) == len(idx_out)), "Expect idx_in and idx_out to be aligned consecutively (x[t] -> y[t-1])."

        self.idx_in = np.array(idx_in, dtype=int)
        self.idx_out = np.array(idx_out, dtype=int)

        # Build (n_case, t_start) pairs for all windows, then split chronologically
        num_windows = len(idx_in) - self.L + 1
        pairs = [(n, k) for n in range(self.N) for k in range(num_windows)]

        # chronological split by k
        import math
        val_count = max(1, math.ceil(val_percent * num_windows))
        split_k = num_windows - val_count   # windows [0..split_k-1] for train; [split_k..] val
        if split_k <= 0:
            raise RuntimeError(
                f"val_percent={val_percent} is too high for num_windows={num_windows} "
                f"(train windows would be {split_k} ≤ 0). Reduce val_percent or increase data length."
            )
        self.train_index = [(n, k) for (n, k) in pairs if k < split_k]
        self.val_index   = [(n, k) for (n, k) in pairs if k >= split_k]
        self.view = "train"
        # Lazily opened handles
        self._ds = None
        self._var = None

    def _open(self):
        # Lazily open per-process (safe with num_workers>0)
        if self._ds is None:
            self._ds = nc.Dataset(self.data_path, "r")
            self._var = self._ds.variables[self.var_name]

    def set_view(self, view):
        assert view in ("train", "val")
        self.view = view

    def __len__(self):
        return len(self.train_index) if self.view == "train" else len(self.val_index)
    
    def __getitem__(self, idx):
        self._open()
        index = self.train_index if self.view == "train" else self.val_index
        n, k = index[idx]

        # Build absolute time indices for this window
        t_in = self.idx_in[k : k + self.L]                # length L
        t_out = self.idx_out[k : k + self.n_unroll]       # length n_unroll

        # Slice: (N, T, C, H, W)
        # window k uses x[:, k : k+L] -> inputs, and y[:, k : k+n_unroll] -> targets
        x = self._var[n, t_in, :self.C_in, :, :]          # (L, C_in, H, W)
        y = self._var[n, t_out, :, :, :]                  # (n_unroll, C_out, H, W)

        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        # --- (1) Remove pole reference per-time, per-channel, for both x and y ---
        if self.remove_pole:
            # Reference at (lat=-1, lon=0)
            # x_ref: (L, C_in)
            x_ref_np = self._var[n, t_in,  :self.C_in, -1, 0].astype(np.float32)
            x_ref = torch.from_numpy(x_ref_np).view(self.L, self.C_in, 1, 1)
            x = x - x_ref

            # y_ref: (L-1, C_out)
            y_ref_np = self._var[n, t_out, :, -1, 0].astype(np.float32)
            y_ref = torch.from_numpy(y_ref_np).view(self.n_unroll, self.C_out, 1, 1)
            y = y - y_ref
        
        # --- (2) Scale by cell area if provided (broadcast over time & channels) ---
        if self.cell_area is not None:
            # self.cell_area is (1,1,H,W)
            x = x * self.cell_area          # -> (L,   C_in,  H, W)
            y = y * self.cell_area          # -> (L-1, C_out, H, W)

        # --- (3) Residual target, after identical preprocessing of x and y ---
        if self.pred_residual:
            # y[..., :C_in] := y - x_prev  (work in CPU, no in-place on saved tensors)
            # x_prev is x at t_in[1:] aligned with y at t_out[:]
            x_prev = x[0:self.L-1, :, :, :]               # (L-1, C_in, H, W)
            y = y.clone()
            y[:, :self.C_in, :, :] = y[:, :self.C_in, :, :] - x_prev

        return x, y
    
    def __del__(self):
        if getattr(self, "_ds", None) is not None:
            try:
                self._ds.close()
            except Exception:
                pass


        
