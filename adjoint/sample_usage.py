import torch
from torch.utils.data import DataLoader
import numpy as np
import xarray as xr
import random
import sys
import os

# add OceanAdjoint to path
sys.path.append("/nobackup/smousav2/adjoint_learning/SSH_only/OceanAdjoint/adjoint")
import model
import data_loaders

# Set global seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


labels = None
C_in = 1
C_out = 1
pred_residual = True
data_path = "/nobackupp17/ifenty/AD_ML/2025-07-25/etan_ad_20250725.nc"
wet_mask_path = "/nobackupp17/ifenty/AD_ML/sam_grid/SAM_GRID_v01.nc"
idx_in_train = [3]
idx_out_train = [6]
idx_in_test = [6]
idx_out_test = [9]

# load wet mask
wet_mask_loader = data_loaders.WetMaskFromNetCDF(
    wet_path=wet_mask_path,
    var_name='wet_mask',
    device="cuda",
    engine="netcdf4"
)
wet = wet_mask_loader.get_wet_mask()  # Shape: (H, W)


# load data
loader = data_loaders.AdjointDatasetFromNetCDF(
    data_path=data_path,
    var_name='etan_ad',
    C_in=C_in,
    idx_in_train=idx_in_train,
    idx_out_train=idx_out_train,
    idx_in_test=idx_in_test,
    idx_out_test=idx_out_test,
    label=None,
    device="cuda"
)

train_ds, test_ds = loader.get_datasets()
train_norm, test_norm = loader.get_norms()

# DataLoader with reproducible shuffling
g = torch.Generator()
g.manual_seed(seed)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, generator=g, num_workers=0)
test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

# Get first batch of data to infer H, W
sample_x, sample_y = train_ds[0]  # (C, H, W)
_, H, W = sample_x.shape

# Create label embedding
embed_dim = 8
embedder = model.CostFunctionEmbedding(enc_dim=C_in, embed_dim=embed_dim, spatial_shape=(H, W))

# Initialize model
if labels is not None:
    model_adj = model.AdjointModel(backbone=model.AdjointNet(wet, in_channels=C_in+embed_dim, out_channels=C_out), pred_residual=pred_residual)
    optimizer = torch.optim.AdamW(list(model_adj.parameters()) + list(embedder.parameters()), lr=1e-4, weight_decay=1e-5)
else:
    model_adj = model.AdjointModel(backbone=model.AdjointNet(wet, in_channels=C_in, out_channels=C_out), pred_residual=pred_residual)
    optimizer = torch.optim.AdamW(model_adj.parameters(), lr=1e-4, weight_decay=1e-5)

# Train the model
model_save_path = "ssh_only_pred_residual.pt"
checkpoint_path = "checkpoint.pt"
start_epoch = 1
best_val_loss = float("inf")
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model_adj.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_val_loss = checkpoint["best_val_loss"]
    start_epoch = checkpoint["epoch"] + 1

    # Restore RNG states for exact reproducibility
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])

    print(f"Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

model.train_adjoint_model(
    model=model_adj,
    dataloader=train_loader,
    optimizer=optimizer,
    val_loader=test_loader,
    num_epochs=5000,
    patience=200,
    label_embedder=embedder,
    save_path=model_save_path,
    start_epoch=start_epoch,
    best_val_loss=best_val_loss,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)