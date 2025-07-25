import model
import data_loaders
import torch
from torch.utils.data import DataLoader
import numpy as np

labels = None
C_in = 1
C_out = 1
pred_residual = True

# load wet mask
wet = None

# Example:
loader = data_loaders.AdjointDatasetFromNetCDF(
    data_path="data.nc",
    var_name="SSH",
    label_name=None,
    test_frac=0.2,
    device="cuda"
)

train_ds, test_ds = loader.get_datasets()
train_norm, test_norm = loader.get_norms()

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)

# Get first batch of data to infer H, W
sample_x, sample_y = train_ds[0]  # (C, H, W)
C_in, H, W = sample_x.shape[-2], sample_x.shape[-1]
C_out = sample_y.shape[0] 
embed_dim = 8
embedder = model.CostFunctionEmbedding(enc_dim=C_in, embed_dim=embed_dim, spatial_shape=(H, W))

if labels is not None:
    model_adj = model.AdjointModel(backbone=model.AdjointNet(wet, in_channels=C_in+embed_dim, out_channels=C_out), pred_residual=pred_residual)
    optimizer = torch.optim.AdamW(list(model_adj.parameters()) + list(embedder.parameters()), lr=1e-4, weight_decay=1e-5)
else:
    model_adj = model.AdjointModel(backbone=model.AdjointNet(wet, in_channels=C_in, out_channels=C_out), pred_residual=pred_residual)
    optimizer = torch.optim.AdamW(model_adj.parameters(), lr=1e-4, weight_decay=1e-5)