import torch
import torch.nn as nn
import numpy as np
from itertools import tee

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class CappedGELU(torch.nn.Module):
    """
    Implements a GeLU with capped maximum value.
    """

    def __init__(self, cap_value=1.0, **kwargs):
        """
        :param cap_value: float: value at which to clip activation
        :param kwargs: passed to torch.nn.LeadyReLU
        """
        super().__init__()
        self.add_module("gelu", torch.nn.GELU(**kwargs))
        self.register_buffer("cap", torch.tensor(cap_value, dtype=torch.float32))

    def forward(self, inputs):
        x = self.gelu(inputs)
        x = torch.clamp(x, max=self.cap)
        return x

class BilinearUpsample(torch.nn.Module):
    def __init__(self, upsampling: int = 2, **kwargs):
        super().__init__()
        self.upsampler = torch.nn.Upsample(scale_factor=upsampling, mode="bilinear")

    def forward(self, x):
        return self.upsampler(x)


class AvgPool(torch.nn.Module):
    def __init__(
        self,
        pooling: int = 2,
    ):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(pooling)

    def forward(self, x):
        return self.avgpool(x)


class ConvNeXtBlock(torch.nn.Module):
    """
    A convolution block as reported in https://github.com/CognitiveModeling/dlwp-hpx/blob/main/src/dlwp-hpx/dlwp/model/modules/blocks.py.

    This is a modified version of the actual ConvNextblock which is used in the HealPix paper.

    """

    def __init__(
        self,
        in_channels: int = 10, # Input channels
        out_channels: int = 4,  # Output channels
        kernel_size: int = 3,
        dilation: int = 1,
        n_layers: int = 1,
        activation: torch.nn.Module = CappedGELU,
        pad="circular",
        upscale_factor: int = 4
    ):
        super().__init__()
        assert kernel_size % 2 != 0, "Cannot use even kernel sizes!"

        self.N_in = in_channels
        self.N_pad = int((kernel_size + (kernel_size - 1) * (dilation - 1) - 1) / 2)  # Padding size
        self.pad = pad

        assert n_layers == 1, "Can only use a single layer here!"

        # 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = torch.nn.Conv2d( # If input and output channels don’t match, it maps input to output shape using a 1×1 convolution.
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
            )

        # Convolution block
        convblock = []
        convblock.append(
            torch.nn.Conv2d(    # Increases the feature dimension (upscale_factor = 4 by default), captures spatial patterns in the input.
                in_channels=in_channels,
                out_channels=int(in_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )
        convblock.append(torch.nn.BatchNorm2d(in_channels * upscale_factor))    # Normalize the output.
        
        convblock.append(activation())
            
        convblock.append(
            torch.nn.Conv2d(    # Another spatial convolution, but keeping the same channel depth.
                in_channels=int(in_channels * upscale_factor),
                out_channels=int(in_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
            )
        )
        convblock.append(torch.nn.BatchNorm2d(in_channels * upscale_factor))

        convblock.append(activation())
            
        # Linear postprocessing
        convblock.append(
            torch.nn.Conv2d(    # Final convolution to map the feature dimension back to the output channels.
                in_channels=int(in_channels * upscale_factor),
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
            )
        )
        self.convblock = torch.nn.Sequential(*convblock)

    def forward(self, x):
        skip = self.skip_module(x)
        for l in self.convblock:
            if isinstance(l, nn.Conv2d) and l.kernel_size[0] != 1:
                x = torch.nn.functional.pad(
                    x, (self.N_pad, self.N_pad, 0, 0), mode=self.pad
                )
                x = torch.nn.functional.pad(
                    x, (0, 0, self.N_pad, self.N_pad), mode="constant"
                )
            x = l(x)
        return skip + x     # This is the residual connection, which adds the input to the output of the convolution block.


class CostFunctionEmbedding(nn.Module):
    def __init__(self, enc_dim, embed_dim, spatial_shape):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(enc_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
            nn.ReLU()
        )
        self.spatial_shape = spatial_shape  # (H, W)
        self.enc_dim = enc_dim

    def forward(self, onehot_label, batch_size):
        """
        onehot_label: (B, enc_dim) — one-hot encoding of cost function J
        Returns: (B, embed_dim, H, W)
        """
        embedding = self.mlp(onehot_label)              # (B, embed_dim)
        embedding = embedding[:, :, None, None]         # → (B, embed_dim, 1, 1)
        embedding = embedding.expand(-1, -1, *self.spatial_shape)  # → (B, embed_dim, H, W)
        return embedding


def adjoint_rollout_loss(model, lam_targets):
    """
    Compute rollout loss over τ steps.
    
    Args:
        model: AdjointModel
        lam_targets: [B, τ+1, C_l, H, W] — ground-truth λ from T to T−τ
    
    Returns:
        loss: scalar
    """
    lam_pred = model(lam_targets)  # [B, τ+1, C_l, H, W]
    return torch.nn.functional.mse_loss(lam_pred, lam_targets)


def train_adjoint_model(
        model, 
        dataloader, 
        optimizer, 
        label_embedder=None,
        device="cuda", 
        num_epochs=50, 
        scheduler=None, 
        log_every=1,
        val_loader=None,
        early_stopping=True,
        patience=5,
        save_path=None
        ):
    model.to(device)
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve_count = 0

    use_labels = True if label_embedder is not None else False

    if use_labels:
        for epoch in range(1, num_epochs + 1):
            # === Training ===
            model.train()
            total_loss = 0.0

            for xb, yb, labelb in dataloader:
                print("check training")
                xb = xb.to(device)  # shape: (B, C_in, H, W)
                yb = yb.to(device)  # shape: (B, C_out, H, W)
                labelb = labelb.to(device)  # shape: (B, C_in)

                # Get one-hot encoding of labels
                onehot = labelb.float()  #torch.nn.functional.one_hot(labelb, num_classes=label_embedder.enc_dim).float()  # (B, C_in)
                embed = label_embedder(onehot, batch_size=xb.shape[0])  # (B, embed_dim, H, W)
                xb = torch.cat([xb, embed], dim=1)  # (B, C_in+embed_dim, H, W)

                optimizer.zero_grad()
                pred = model(xb)
                loss = torch.nn.functional.mse_loss(pred, yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(dataloader)

            # === Validation ===
            avg_val_loss = None
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb, labelb in val_loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        labelb = labelb.to(device)
                        onehot = labelb.float()  #torch.nn.functional.one_hot(labelb, num_classes=label_embedder.enc_dim).float()  # (B, C_in)
                        embed = label_embedder(onehot, batch_size=xb.shape[0])  # (B, embed_dim, H, W)
                        xb = torch.cat([xb, embed], dim=1)  # (B, C_in+embed_dim, H, W)
                        pred = model(xb)
                        loss = torch.nn.functional.mse_loss(pred, yb)
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)

                # Early stopping logic
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    no_improve_count = 0
                    if save_path:
                        torch.save(model.state_dict(), save_path)
                else:
                    no_improve_count += 1
                    if early_stopping and no_improve_count >= patience:
                        print(f"Early stopping at epoch {epoch} (best at epoch {best_epoch}, val_loss={best_val_loss:.6f})")
                        break


            if scheduler is not None:
                scheduler.step()
        
            if epoch % log_every == 0:
                msg = f"[Epoch {epoch:03d}] Train Loss: {avg_train_loss:.6f}"
                if avg_val_loss is not None:
                    msg += f" | Val Loss: {avg_val_loss:.6f}"
                print(msg)

    else:
        for epoch in range(1, num_epochs + 1):
            # === Training ===
            model.train()
            total_loss = 0.0

            for xb, yb in dataloader:
                print("check training")
                xb = xb.to(device)  # shape: (B, C_in, H, W)
                yb = yb.to(device)  # shape: (B, C_out, H, W)

                optimizer.zero_grad()
                pred = model(xb)
                loss = torch.nn.functional.mse_loss(pred, yb)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(dataloader)

            # === Validation ===
            avg_val_loss = None
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        pred = model(xb)
                        loss = torch.nn.functional.mse_loss(pred, yb)
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)

                # Early stopping logic
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_epoch = epoch
                    no_improve_count = 0
                    if save_path:
                        torch.save(model.state_dict(), save_path)
                else:
                    no_improve_count += 1
                    if early_stopping and no_improve_count >= patience:
                        print(f"Early stopping at epoch {epoch} (best at epoch {best_epoch}, val_loss={best_val_loss:.6f})")
                        break


            if scheduler is not None:
                scheduler.step()
        
            if epoch % log_every == 0:
                msg = f"[Epoch {epoch:03d}] Train Loss: {avg_train_loss:.6f}"
                if avg_val_loss is not None:
                    msg += f" | Val Loss: {avg_val_loss:.6f}"
                print(msg)

        

class BaseAdjointNet(torch.nn.Module):
    def __init__(self, ch_width, last_kernel_size, pad):
        super().__init__()
        assert last_kernel_size % 2 != 0, "Cannot use even kernel sizes!"
        self.ch_width = ch_width
        self.N_in = ch_width[0]
        self.N_out = ch_width[-1]
        self.N_pad = (last_kernel_size - 1) // 2
        self.pad = pad

    def forward_once(self, x):
        raise NotImplementedError()


class AdjointNet(BaseAdjointNet):
    def __init__(
        self,
        wet,
        in_channels,   # = C_in
        ch_width=[64, 128, 256, 512],
        out_channels=4,  # C_out
        core_block=ConvNeXtBlock,
        down_sampling_block=AvgPool,
        up_sampling_block=BilinearUpsample,
        activation=CappedGELU,
        dilation=[1, 2, 4, 8],
        n_layers=[1, 1, 1, 1],
        last_kernel_size=3,
        pad="circular",
    ):
        super().__init__(ch_width, last_kernel_size, pad)
        self.register_buffer("wet", wet)
        ch_width = [in_channels] + ch_width
        self.num_steps = len(ch_width) - 1

        layers = []
        for i, (a, b) in enumerate(pairwise(ch_width)):
            layers.append(core_block(a, b, dilation=dilation[i], n_layers=n_layers[i], activation=activation, pad=pad))
            layers.append(down_sampling_block())

        layers.append(core_block(b, b, dilation=dilation[-1], n_layers=n_layers[-1], activation=activation, pad=pad))
        layers.append(up_sampling_block(in_channels=b, out_channels=b))

        ch_width.reverse()
        dilation.reverse()
        n_layers.reverse()
        for i, (a, b) in enumerate(pairwise(ch_width[:-1])):
            layers.append(core_block(a, b, dilation=dilation[i], n_layers=n_layers[i], activation=activation, pad=pad))
            layers.append(up_sampling_block(in_channels=b, out_channels=b))

        layers.append(core_block(b, b, dilation=dilation[0], n_layers=n_layers[0], activation=activation, pad=pad))
        layers.append(nn.Conv2d(b, out_channels, last_kernel_size))

        self.layers = nn.ModuleList(layers)

    def forward_once(self, fts):    # fts: [B, C_in, H, W]
        temp = [None] * self.num_steps
        count = 0
        for l in self.layers:
            if isinstance(l, nn.Conv2d) and l.kernel_size[0] != 1:
                fts = torch.nn.functional.pad(fts, (self.N_pad, self.N_pad, 0, 0), mode=self.pad)
                fts = torch.nn.functional.pad(fts, (0, 0, self.N_pad, self.N_pad), mode="constant")
            fts = l(fts)
            if count < self.num_steps and isinstance(l, ConvNeXtBlock):
                temp[count] = fts
                count += 1
            elif count >= self.num_steps and isinstance(l, BilinearUpsample):
                crop = np.array(fts.shape[2:])
                target = np.array(temp[2 * self.num_steps - count - 1].shape[2:])
                pads = target - crop
                pads = [pads[1] // 2, pads[1] - pads[1] // 2, pads[0] // 2, pads[0] - pads[0] // 2]
                fts = nn.functional.pad(fts, pads)
                fts += temp[2 * self.num_steps - count - 1]
                count += 1
        return torch.mul(fts, self.wet)  # shape: [B, C_out, H, W]



class AdjointModel(torch.nn.Module):
    def __init__(self, backbone, pred_residual=False):
        super().__init__()
        self.model = backbone
        self.pred_residual = pred_residual

    def forward(self, x_true):
        """
        Inputs:
            x_true:       [B, C_in+embed_dim, H, W]
        Returns:
            preds:       [B, C_out, H, W]
        """
        pred = self.model.forward_once(x_true)  # [B, C_out, H, W]
        if self.pred_residual:
            pred = pred + x_true[:, :pred.shape[1]]  # add residual to matching input channels
        return pred




### Modify this function for your use case
def generate_adjoint_rollout(model, x_seq_true, wet=None, train=False):
    """
    Run backward adjoint rollout using trained model.
    
    Args:
        model: trained AdjointModel
        x_seq_true: [B, T, C_in, H, W] or [T, C_in, H, W] -> ordered [λ(T), λ(T−1), ..., λ(T−τ)]
        wet: optional [H, W] mask
        train: bool
    
    Returns:
        y_seq: [B, T, C_out, H, W] or [T, C_out, H, W] - from λ(T) to λ(T−τ)
    """
    device = next(model.parameters()).device
    model.eval()

    # Add batch dim if missing
    added_batch_dim = False
    if x_seq_true.ndim == 4:
        x_seq_true = x_seq_true.unsqueeze(0)  # → [1, T, C_in, H, W]
        added_batch_dim = True

    x_seq_true = x_seq_true.to(device)
    B, T, C_in, H, W = x_seq_true.shape
    preds = []
    x_t_plus_1 = x_seq_true[:, 0]  # λ(T)

    with torch.no_grad():
        for t in range(T):
            y_t = model(x_t_plus_1) 
            if wet is not None:
                y_t = y_t * wet.to(device)

            preds.append(y_t)
            x_t_plus_1 = y_t

    y_seq = torch.stack(preds, dim=1)  # shape: [B, T, C_out, H, W]

    if added_batch_dim:
        y_seq = y_seq.squeeze(0)  # return to [T, C_out, H, W]

    return y_seq.cpu()