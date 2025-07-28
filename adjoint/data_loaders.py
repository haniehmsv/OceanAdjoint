import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

class AdjointDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, labels=None):
        """
        Args:
            x: tensor of shape (N_samples, T, 2, H, W) – input adjoints, note that N_samples is the number of spatial sample points * 2 (one for SSH and one for OBP)
            y: tensor of shape (N_samples, T, 4, H, W) – output adjoints
            labels: tensor of shape (N_samples,) with values 0 (OBP) or 1 (SSH)
        """
        super().__init__()
        self.N, self.T, self.C_in, self.H, self.W = x.shape
        self.labels = labels
        # self.label_embedder = label_embedder

        # Store output
        self.y_flat = y.reshape(-1, y.shape[2], self.H, self.W)  # (N*T, 4, H, W)

        # --- Efficient label embedding ---
        # onehot_labels = torch.nn.functional.one_hot(labels, num_classes=2).float()  # (N, 2)
        # with torch.no_grad():
        #     label_embed = self.label_embedder(onehot_labels, batch_size=self.N) # (N, embed_dim, H, W)

        # Expand to temporal shape: (N, T, embed_dim, H, W)
        # label_embed = label_embed.unsqueeze(1).expand(-1, self.T, -1, -1, -1)

        # Concatenate along channel dimension: (N, T, C_in + embed_dim, H, W)
        # x_with_label = torch.cat([x, label_embed], dim=2)

        # Flatten temporal dimension: (N*T, C_in + embed_dim, H, W)
        self.x_flat = x.reshape(-1, x.shape[2], self.H, self.W)

        # Flatten labels to match x_flat: repeat each label T times
        if self.labels is not None:
            self.label_flat = labels.repeat_interleave(self.T)   # (N*T,)
        else:
            self.label_flat = None

    def __len__(self):
        return self.x_flat.shape[0]

    def __getitem__(self, idx):
        if self.label_flat is not None:
            return self.x_flat[idx], self.y_flat[idx], self.label_flat[idx]
        else:
            return self.x_flat[idx], self.y_flat[idx]



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
                 device="cpu",               # Optional torch device
                 engine="netcdf4"            # Engine to use for reading NetCDF
                ):
        self.device = device

        # Load the NetCDF file
        ds = xr.open_dataset(data_path, engine=engine)
        data = ds[var_name].values            # Shape: (N_targets, T, C, H, W)
        data = torch.tensor(data, dtype=torch.float32).to(device)

        N, T, C_out, H, W = data.shape
        x_train = data[:, idx_in_train, :C_in]                      # (N, T_train, C_in, H, W)
        y_train = data[:, idx_out_train]                            # (N, T_train, C_out, H, W)
        x_test  = data[:, idx_in_test, :C_in]                       # (N, T_test, C_in, H, W)
        y_test  = data[:, idx_out_test]                             # (N, T_test, C_out, H, W)


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
