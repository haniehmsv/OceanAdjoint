### Modify these functions for your use case
import torch
import time, psutil, os

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
    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_mem_cpu = process.memory_info().rss / 1e9   # GB
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ---------------------------
    device = next(model.parameters()).device
    wet_gpu = wet.to(device, dtype=torch.float32)
    if remove_pole:
        wet_gpu[-1, 0] = 0.0

    model.eval()

    if x_seq_true.is_cuda: x_seq_true = x_seq_true.cpu()

    # Add batch dim if missing
    added_batch_dim = False
    if x_seq_true.ndim == 4:
        x_seq_true = x_seq_true.unsqueeze(0)  # → [1, T, C_in, H, W]
        added_batch_dim = True

    B, T, C_in, H, W = x_seq_true.shape
    data_mean = data_mean.to(device)
    data_std  = data_std.to(device)
    y_seq_true_cpu = x_seq_true[:, 1:] * wet  # λ(T-1) to λ(T-τ) shape: [B, T-1, C_in, H, W]
    y_seq_pred_cpu = torch.empty((B, T-1, C_in, H, W), dtype=torch.float32, device="cpu")

    def to_std_gpu(x_cpu_slice):  # x_cpu_slice: [B,C_in,H,W] on CPU
        xg = x_cpu_slice.to(device, non_blocking=True)
        return (xg - data_mean) / data_std  # [B,C_in,H,W] on GPU
    
    input_std = to_std_gpu(x_seq_true[:,0])  # [B,C_in,H,W] on GPU

    # Rollout
    for i in range(T-1):
        y_t_std = model(input_std)  # shape: [B, C_in, H, W]
        if pred_residual:
            y_t_std = y_t_std + input_std

        if wet_gpu is not None:
            y_t_std *= wet_gpu

        y_t = y_t_std * data_std + data_mean  # unnormalize
        y_seq_pred_cpu[:, i] = y_t.detach().to("cpu")

        input_std = y_t_std.detach()    # stays on GPU

        del y_t_std, y_t
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if added_batch_dim:
        y_seq_pred_cpu = y_seq_pred_cpu.squeeze(0)  # return to [T, C_out, H, W]
        y_seq_true_cpu = y_seq_true_cpu.squeeze(0)  # return to [T-1, C_in, H, W]

    if cell_area is not None:
        cell_area = cell_area.to("cpu")
        y_seq_true_cpu /= cell_area.view(1, 1, H, W)
        y_seq_pred_cpu /= cell_area.view(1, 1, H, W)
    
    ### --- Report monitoring stats ---
    elapsed = time.time() - start_time
    end_mem_cpu = process.memory_info().rss / 1e9
    print(f"Rollout time: {elapsed:.2f} seconds")
    print(f"CPU memory change: {end_mem_cpu - start_mem_cpu:+.2f} GB (total now {end_mem_cpu:.2f} GB)")
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    return y_seq_true_cpu, y_seq_pred_cpu