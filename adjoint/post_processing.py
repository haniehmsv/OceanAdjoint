### Modify these functions for your use case
import torch

def normalization_constants(input):
    """
    calculates normalization constants for the input sequence.
    
    Args:
        input: [B, C_in, H, W]     # True adjoint sequence

    Returns:
        x_norms: [B,] # Normalization constants
    """
    norms = input.abs().amax(dim=(1, 2, 3))  # Maximum absolute value over channel, height, width, shape: [B,]

    return norms


def generate_adjoint_rollout(model, x_seq_true, wet=None, pred_residual=False, cell_area=None):
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
    model.eval()

    # Add batch dim if missing
    added_batch_dim = False
    if x_seq_true.ndim == 4:
        x_seq_true = x_seq_true.unsqueeze(0)  # → [1, T, C_in, H, W]
        added_batch_dim = True

    y_seq_true = x_seq_true[:, 1:] * wet  # λ(T-1) to λ(T-τ) shape: [B, T-1, C_in, H, W]

    x_seq_true = x_seq_true.to(device)
    B, T, C_in, H, W = x_seq_true.shape
    preds = []
    input = x_seq_true[:, 0].clone()  # λ(T) shape: [B, C_in, H, W]
    norms = normalization_constants(input).view(-1, 1, 1, 1)
    input /= norms  # Normalize input by its norm

    with torch.no_grad():
        for t in range(T-1):
            y_t = model(input)  # shape: [B, C_out, H, W]
            if pred_residual:
                y_t[:, :C_in] = y_t[:, :C_in] + input
            y_t *= norms
            if wet is not None:
                y_t *= wet.to(device)

            preds.append(y_t)
            input = y_t[:,:C_in].clone()
            norms = normalization_constants(input).view(-1, 1, 1, 1)
            input /= norms

    y_seq = torch.stack(preds, dim=1)  # shape: [B, T-1, C_out, H, W]

    if added_batch_dim:
        y_seq = y_seq.squeeze(0)  # return to [T, C_out, H, W]
        y_seq_true = y_seq_true.squeeze(0)  # return to [T-1, C_in, H, W]

    if cell_area is not None:
        cell_area = cell_area.to(device)
        y_seq_true /= cell_area.view(1, 1, H, W)
        y_seq /= cell_area.view(1, 1, H, W)
    return y_seq_true.cpu(), y_seq.cpu()