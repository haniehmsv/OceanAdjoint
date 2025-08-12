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
data_path = "/nobackupp17/ifenty/AD_ML/2025-07-28/etan_ad_20250728b_combined.nc"
wet_mask_path = "/nobackupp17/ifenty/AD_ML/sam_grid/SAM_GRID_v01.nc"
idx_in = [3,4,5,6,7,8,9]
idx_out = [4,5,6,7,8,9]
n_unroll = 4
n_epochs = 1000

# === Distributed init ===
device, local_rank = init_distributed_mode()

# load wet mask
wet_mask_loader = data_loaders.WetMaskFromNetCDF(
    wet_path=wet_mask_path,
    var_name='wet_mask',
    device=device,
    engine="netcdf4"
)
wet = wet_mask_loader.get_wet_mask()  # Shape: (H, W)

# load cell area
cell_area_loader = data_loaders.WetMaskFromNetCDF(
    wet_path=wet_mask_path,
    var_name='area',
    device=device,
    engine="netcdf4"
)
cell_area = cell_area_loader.get_wet_mask()  # Shape: (H, W)
area_weighting = cell_area/cell_area.max()


# load data
loader = data_loaders.AdjointRolloutDatasetFromNetCDF(
    data_path=data_path,
    var_name='etan_ad',
    C_in=C_in,
    idx_in=idx_in,
    idx_out=idx_out,
    n_unroll=n_unroll,
    pred_residual=pred_residual,
    device=device
)

train_ds, test_ds = loader.get_datasets()
train_loader, test_loader, train_sampler, test_sampler = data_loaders.get_distributed_loaders(
    train_ds, test_ds, batch_size=16, num_workers=4
)

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
train_norm, test_norm = loader.get_norms()

# DataLoader with reproducible shuffling
g = torch.Generator()
g.manual_seed(seed)


# Get first batch of data to infer H, W
sample_x, sample_y = train_ds[0]  # (L, C, H, W)
_, _, H, W = sample_x.shape

# Create label embedding
# embed_dim = 8
# embedder = model.CostFunctionEmbedding(enc_dim=C_in, embed_dim=embed_dim, spatial_shape=(H, W))

# Initialize model
world_size = dist.get_world_size()
model_adj = model.AdjointModel(backbone=model.AdjointNet(wet, in_channels=C_in, out_channels=C_out)).to(device)
optimizer = torch.optim.AdamW(model_adj.parameters(), lr=1e-4, weight_decay=1e-5)

model_adj = DDP(model_adj, device_ids=[local_rank])

# scheduler = CosineAnnealingLR(optimizer,T_max=n_epochs, eta_min=0.0)
scheduler = None

# Train the model
checkpoint_path = "checkpoints/checkpoint_all_data_all_pair_one_step_interval.pt"
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
    label_embedder=None,
    checkpoint_path=checkpoint_path,
    start_epoch=start_epoch,
    best_val_loss=best_val_loss,
    device=device,
    area_weighting=None
)
dist.destroy_process_group()