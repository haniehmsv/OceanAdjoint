import torch
from torch.utils.data import DataLoader
import numpy as np
import xarray as xr
import random
import sys
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# add OceanAdjoint to path
sys.path.append("/nobackup/smousav2/adjoint_learning/SSH_only_rollout_loss/OceanAdjoint/adjoint")
import model
import data_loaders

# Set global seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

def init_distributed_mode():
    dist.init_process_group(
        backend="nccl",   # use "gloo" if CPUs
        init_method="env://"
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank), local_rank

# === Parameters ===
C_in = 1
C_out = 1
pred_residual = True
remove_pole = True
data_path = "/nobackupp17/ifenty/AD_ML/2025-08-05b/all_adetan_training_points/consolidated/etan_ad_2025-08-05b_3594_consolidated.nc"
wet_mask_path = "/nobackupp17/ifenty/AD_ML/sam_grid/SAM_GRID_v01.nc"
idx_in = [3,4,5,6,7,8]
idx_out = [4,5,6,7,8,9]
n_unroll = 3
n_epochs = 1000
val_percent = 0.1

# === Distributed init ===
device, local_rank = init_distributed_mode()

# load wet mask
wet_mask_loader = data_loaders.WetMaskFromNetCDF(
    wet_path=wet_mask_path,
    var_name='wet_mask',
    engine="netcdf4"
)
wet = wet_mask_loader.get_wet_mask()  # Shape: (H, W)

# load cell area
cell_area_loader = data_loaders.WetMaskFromNetCDF(
    wet_path=wet_mask_path,
    var_name='area',
    engine="netcdf4"
)
cell_area = cell_area_loader.get_wet_mask()  # Shape: (H, W)
area_weighting = (cell_area/cell_area.max()).to(device)

# DataLoader with reproducible shuffling
g = torch.Generator()
g.manual_seed(seed)

# load data
loader = data_loaders.AdjointRolloutDatasetFromNetCDF(
    data_path=data_path,
    var_name='etan_ad',
    C_in=C_in,
    idx_in=idx_in,
    idx_out=idx_out,
    n_unroll=n_unroll,
    pred_residual=pred_residual,
    remove_pole=remove_pole,
    val_percent=val_percent
)

train_ds, test_ds = loader.get_datasets()
train_loader, _, train_sampler, _ = data_loaders.get_distributed_loaders(
    train_ds, test_ds, batch_size=16, num_workers=4, generator=g, pin_memory=True
)

# save data stats
data_mean, data_std = loader.get_mean_std()
if (not dist.is_initialized()) or dist.get_rank() == 0:
    norm_path = f"data_norm_sequence_of_{n_unroll}.npz"
    if not os.path.exists(norm_path):
        np.savez(
            norm_path,
            mean=(data_mean.detach().cpu().numpy()
                  if isinstance(data_mean, torch.Tensor) else np.asarray(data_mean)),
            std=(data_std.detach().cpu().numpy()
                 if isinstance(data_std, torch.Tensor) else np.asarray(data_std)),
        )
        print(f"[Rank 0] Saved normalization stats → {norm_path}")
    else:
        print(f"[Rank 0] Normalization stats already exist → {norm_path}")


if dist.get_rank() == 0:  # only run validation on rank 0
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
else:
    test_loader = None


# Get first batch of data to infer H, W
sample_x, sample_y = train_ds[0]  
_, _, H, W = sample_x.shape     # (n_unroll, C, H, W)

# Initialize model
ckpt = torch.load("/nobackup/smousav2/adjoint_learning/SSH_only_weighted_loss/checkpoints/checkpoint_all_data_all_pair_one_step_interval.pt", map_location="cpu")
state = ckpt["model_state_dict"]
model_adj = model.AdjointModel(backbone=model.AdjointNet(wet, in_channels=C_in, out_channels=C_out)).to(device)
missing, unexpected = model_adj.load_state_dict(state, strict=False)
if dist.get_rank() == 0:
    print("Transfer load: missing keys:", missing)
    print("Transfer load: unexpected keys:", unexpected)

optimizer = torch.optim.AdamW(model_adj.parameters(), lr=1e-4, weight_decay=1e-5)
model_adj = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_adj)
model_adj = DDP(model_adj, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

# scheduler = CosineAnnealingLR(optimizer,T_max=n_epochs, eta_min=0.0)
scheduler = None

# Train the model
checkpoint_path = f"checkpoints/checkpoint_sequence_of_{n_unroll}.pt"
start_epoch = 1
best_val_loss = float("inf")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_to_load = model_adj.module if hasattr(model_adj, "module") else model_adj
    model_to_load.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    best_val_loss = checkpoint["best_val_loss"]
    start_epoch = checkpoint["epoch"] + 1
    
    print(f"Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

model.train_adjoint_model(
    model=model_adj,
    dataloader=train_loader,
    optimizer=optimizer,
    val_loader=test_loader,
    num_epochs=n_epochs,
    scheduler=scheduler,
    patience=20,
    checkpoint_path=checkpoint_path,
    start_epoch=start_epoch,
    best_val_loss=best_val_loss,
    device=device,
    pred_residual=pred_residual,
    area_weighting=area_weighting
)
dist.destroy_process_group()