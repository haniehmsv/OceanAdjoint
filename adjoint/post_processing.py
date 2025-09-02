### Modify these functions for your use case
import torch

@torch.no_grad()
def generate_adjoint_rollout(model, x_seq_true, y_seq_true, data_mean, data_std, C_out_total, wet=None, pred_residual=False, remove_pole=False):
    """
    Run backward adjoint rollout using trained model.
    
    Args:
        model: trained AdjointModel
        x_seq_true: [B, T, C_in, H, W] or [T, C_in, H, W] -> ordered [λ(T), λ(T−1), ..., λ(T−τ)] (unnormalized)
        y_seq_true: [B, T, C_out, H, W] or [T, C_out, H, W] -> ordered [f(T), f(T−1), ..., f(T−τ)] (unnormalized)
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
    y_seq_true = y_seq_true.to(device)
    data_mean = data_mean.to(device)
    data_std = data_std.to(device)

    model.eval()

    # Add batch dim if missing
    added_batch_dim = False
    if x_seq_true.ndim == 4:
        x_seq_true = x_seq_true.unsqueeze(0)  # → [1, T, C_in, H, W]
        added_batch_dim = True

    B, T, C_in, H, W = x_seq_true.shape
    _, _, C_out, _, _ = y_seq_true.shape

    if C_out_total > C_out:
        out_seq_true = torch.cat([x_seq_true[:,1:], y_seq_true[:,1:]], dim=2)
    else:
        out_seq_true = y_seq_true[:,1:]

    mean = data_mean.view(1, 3, H, W)
    std = data_std.view(1, 3, H, W)
    x_seq_stdized = (x_seq_true - mean[None, :, :C_in, :, :]) / std[None, :, :C_in, :, :]  # standardize input data to the model [B, T, C_in, H, W]

    # standardize input data to the model
    preds = []
    input = x_seq_stdized[:, 0]  # [B, C_in, H, W]

    for i in range(T-1):
        y_t_stdized = model(input)  # shape: [B, C_out_total, H, W]
        if pred_residual:
            y_t_stdized[:, :C_in] = y_t_stdized[:, :C_in] + input

        if wet is not None:
            y_t_stdized *= wet

        preds.append(y_t_stdized.clone())
        if C_out_total > C_out:
            input = y_t_stdized[:, :C_in]  # next input is λ(t)
        else:
            input = x_seq_stdized[:, i+1]

    y_seq = torch.stack(preds, dim=1)  # shape: [B, T-1, C_out, H, W]
    if C_out_total > C_out:
        y_seq = y_seq * std[None,:, :, :, :] + mean[None,:, :, :, :]  # unnormalize
    else:
        y_seq = y_seq * std[None, :, C_in:, :, :] + mean[None, :, C_in:, :, :]  # unnormalize
    
    if added_batch_dim:
        y_seq = y_seq.squeeze(0)  # return to [T, C_out, H, W]
        out_seq_true = out_seq_true.squeeze(0)  # return to [T-1, C_in, H, W]

    return out_seq_true.cpu(), y_seq.cpu()