### Modify these functions for your use case
import torch

@torch.no_grad()
def generate_adjoint_rollout(model, x_seq_true, data_mean, data_std, wet=None, pred_residual=False, remove_pole=False, cell_area=None):
    """
    Run backward adjoint rollout using trained model.
    
    Args:
        model: trained AdjointModel
        x_seq_true: [B, T, C_in, H, W] or [T, C_in, H, W] -> ordered [λ(T), λ(T−1), ..., λ(T−τ)] (unnormalized)
        wet: optional [H, W] mask
    
    Returns:
        y_seq_true: [B, T-1, C_in, H, W] — ground truth (unnormalized)
        y_seq_pred: [B, T-1, C_out, H, W] — predictions (unnormalized)
    """
    device = next(model.parameters()).device
    wet = wet.to(device)
    if remove_pole:
        wet[-1, 0] = 0
    x_seq_true = x_seq_true.to(device)
    data_mean = data_mean.to(device)
    data_std = data_std.to(device)

    model.eval()

    # Add batch dim if missing
    added_batch_dim = False
    if x_seq_true.ndim == 4:
        x_seq_true = x_seq_true.unsqueeze(0)  # → [1, T, C_in, H, W]
        added_batch_dim = True

    y_seq_true = x_seq_true[:, 1:] * wet  # λ(T-1) to λ(T-τ) shape: [B, T-1, C_in, H, W]

    B, T, C_in, H, W = x_seq_true.shape
    mean = data_mean.view(1, C_in, H, W)
    std = data_std.view(1, C_in, H, W)

    # standardize input data to the model
    preds = []
    input = x_seq_true[:, 0].clone()
    input = (input - mean) / std  # λ(T) shape: [B, C_in, H, W]

    for _ in range(T-1):
        y_t_stdized = model(input)  # shape: [B, C_out, H, W]
        if pred_residual:
            y_t_stdized[:, :C_in] = y_t_stdized[:, :C_in] + input

        if wet is not None:
            y_t_stdized *= wet

        y_t = y_t_stdized * std + mean  # unnormalize
        y_t *= wet
        preds.append(y_t.clone())
        input = y_t_stdized

    y_seq = torch.stack(preds, dim=1)  # shape: [B, T-1, C_out, H, W]

    if added_batch_dim:
        y_seq = y_seq.squeeze(0)  # return to [T, C_out, H, W]
        y_seq_true = y_seq_true.squeeze(0)  # return to [T-1, C_in, H, W]

    if cell_area is not None:
        cell_area = cell_area.to(device)
        y_seq_true /= cell_area.view(1, 1, H, W)
        y_seq /= cell_area.view(1, 1, H, W)
    return y_seq_true.cpu(), y_seq.cpu()