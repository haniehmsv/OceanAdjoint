import model
import data_loaders
import torch
from torch.utils.data import DataLoader
import numpy as np
import xarray as xr

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

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)

# Get first batch of data to infer H, W
sample_x, sample_y = train_ds[0]  # (C, H, W)
_, H, W = sample_x.shape[-2], sample_x.shape[-1]

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
model_save_path = "best_model.pt"
model.train_adjoint_model(
    model=model_adj,
    dataloader=train_loader,
    optimizer=optimizer,
    val_loader=test_loader,
    num_epochs=5000,
    patience=200,
    label_embedder=embedder,
    save_path=model_save_path,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)