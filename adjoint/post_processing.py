### Modify these functions for your use case
import torch
import time, psutil, os

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
    if y_seq_true.is_cuda: y_seq_true = y_seq_true.cpu()

    # Add batch dim if missing
    added_batch_dim = False
    if x_seq_true.ndim == 4:
        x_seq_true = x_seq_true.unsqueeze(0)  # → [1, T, C_in, H, W]
        added_batch_dim = True

    B, T, C_in, H, W = x_seq_true.shape
    _, _, C_out, _, _ = y_seq_true.shape

    data_mean = data_mean.to(device)
    data_std  = data_std.to(device)
    C_eff = C_out_total if C_out_total > C_out else C_out
    y_seq_out_cpu = torch.empty((B, T-1, C_eff, H, W), dtype=torch.float32, device="cpu")


    if C_out_total > C_out:
        out_seq_true_cpu = torch.cat([x_seq_true[:,1:], y_seq_true[:,1:]], dim=2)
    else:
        out_seq_true_cpu = y_seq_true[:,1:]
    
    def to_std_gpu(x_cpu_slice):  # x_cpu_slice: [B,C_in,H,W] on CPU
        xg = x_cpu_slice.to(device, non_blocking=True)
        return (xg - data_mean[:C_in]) / data_std[:C_in]  # [B,C_in,H,W] on GPU

    input_std = to_std_gpu(x_seq_true[:,0])  # [B,C_in,H,W] on GPU

    # Rollout
    for i in range(T-1):
        y_t_std = model(input_std)  # shape: [B, C_out_total, H, W]
        if pred_residual:
            y_t_std[:, :C_in] = y_t_std[:, :C_in] + input_std

        if wet_gpu is not None:
            y_t_std *= wet_gpu
        
        if C_out_total > C_out:
            y_t = y_t_std * data_std + data_mean
        else:
            y_t = y_t_std * data_std[C_in:] + data_mean[C_in:]

        y_seq_out_cpu[:, i] = y_t.detach().to("cpu")

        # Next input
        if C_out_total > C_out:
            # Feedback predicted λ (still stdized)
            input_std = y_t_std[:, :C_in].detach()  # stays on GPU
        else:
            # Consume next true λ(t-1) from CPU, standardize on GPU
            input_std = to_std_gpu(x_seq_true[:, i + 1])

        del y_t_std, y_t
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    if added_batch_dim:
        y_seq_out_cpu = y_seq_out_cpu.squeeze(0)  # return to [T, C_out, H, W]
        out_seq_true_cpu = out_seq_true_cpu.squeeze(0)  # return to [T-1, C_in, H, W]
    
    ### --- Report monitoring stats ---
    elapsed = time.time() - start_time
    end_mem_cpu = process.memory_info().rss / 1e9
    print(f"Rollout time: {elapsed:.2f} seconds")
    print(f"CPU memory change: {end_mem_cpu - start_mem_cpu:+.2f} GB (total now {end_mem_cpu:.2f} GB)")
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    return out_seq_true_cpu, y_seq_out_cpu


@torch.no_grad()
def generate_adjoint_rollout_from_initial_lag(model, x0, data_mean, data_std, C_out_total, n_lags, wet=None, pred_residual=False, remove_pole=False):
    """
    Run backward adjoint rollout using trained model.
    
    Args:
        model: trained AdjointModel
        x0: [B, C_in, H, W] or [C_in, H, W] -> the initial unnormalized adjoint state at time t
        data_mean: [C_out_total, H,W] -> mean from training samples used for standardization
        data_std: [C_out_total, H,W] -> std from training samples used for standardization
        C_out_total: int: total number of output channels (states/forcings or states + forcings)
        n_lags: int: number of lagged steps to rollout
        wet: optional [H, W] mask
    
    Returns:
        y_seq_pred: [B, n_lags, C_out, H, W] — predictions (unnormalized)
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

    if x0.is_cuda: x0 = x0.cpu()

    # Add batch dim if missing
    added_batch_dim = False
    if x0.ndim == 4:
        x0 = x0.unsqueeze(0)  # → [1, C_in, H, W]
        added_batch_dim = True

    B, C_in, H, W = x0.shape

    data_mean = data_mean.to(device)
    data_std  = data_std.to(device)
    y_seq_out_cpu = torch.empty((B, n_lags, C_out_total, H, W), dtype=torch.float32, device="cpu")

    
    def to_std_gpu(x_cpu_slice):  # x_cpu_slice: [B,C_in,H,W] on CPU
        xg = x_cpu_slice.to(device, non_blocking=True)
        return (xg - data_mean[:C_in]) / data_std[:C_in]  # [B,C_in,H,W] on GPU

    input_std = to_std_gpu(x0)  # [B,C_in,H,W] on GPU

    # Rollout
    for i in range(n_lags):
        y_t_std = model(input_std)  # shape: [B, C_out_total, H, W]
        if pred_residual:
            y_t_std[:, :C_in] = y_t_std[:, :C_in] + input_std

        if wet_gpu is not None:
            y_t_std *= wet_gpu
        
        y_t = y_t_std * data_std + data_mean    # unnormalize prediction

        y_seq_out_cpu[:, i] = y_t.detach().to("cpu")

        # Next input
        # Feedback predicted λ (still stdized)
        input_std = y_t_std[:, :C_in].detach()  # stays on GPU

        del y_t_std, y_t
        if device.type == "cuda":
            torch.cuda.empty_cache()
    
    if added_batch_dim:
        y_seq_out_cpu = y_seq_out_cpu.squeeze(0)  # return to [n_lags, C_out, H, W]
    
    ### --- Report monitoring stats ---
    elapsed = time.time() - start_time
    end_mem_cpu = process.memory_info().rss / 1e9
    print(f"Rollout time: {elapsed:.2f} seconds")
    print(f"CPU memory change: {end_mem_cpu - start_mem_cpu:+.2f} GB (total now {end_mem_cpu:.2f} GB)")
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    return y_seq_out_cpu



@torch.no_grad()
def generate_adjoint_rollout_with_two_models(model_A, x_seq_true, data_mean_A, data_std_A, pred_residual_A,
                             model_G, y_seq_true, data_mean_G, data_std_G, pred_residual_G,
                             C_out_total, wet=None, remove_pole=False):
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
    device = next(model_A.parameters()).device
    wet = wet.to(device)
    if remove_pole:
        wet[-1, 0] = 0
    x_seq_true = x_seq_true.to(device)
    data_mean_A = data_mean_A.to(device)
    data_std_A = data_std_A.to(device)
    y_seq_true = y_seq_true.to(device)
    data_mean_G = data_mean_G.to(device)
    data_std_G = data_std_G.to(device)

    model_A.eval()
    model_G.eval()

    # Add batch dim if missing
    added_batch_dim = False
    if x_seq_true.ndim == 4:
        x_seq_true = x_seq_true.unsqueeze(0)  # → [1, T, C_in, H, W]
        added_batch_dim = True

    B, T, C_in, H, W = x_seq_true.shape
    _, _, C_out, _, _ = y_seq_true.shape

    if C_out_total > C_out:
        raise RuntimeError(
            f"both states and forcing are predicted in the current implementation."
            f"Use generate_adjoint_rollout instead"
        )

    out_seq_true = y_seq_true[:,1:]

    mean_A = data_mean_A.view(1, C_in, H, W)
    std_A = data_std_A.view(1, C_in, H, W)
    x_seq_stdized = (x_seq_true - data_mean_A[None, None, :, :, :]) / data_std_A[None, None, :, :, :]  # standardize input data to the model [B, T, C_in, H, W]

    mean_G = data_mean_G.view(1, 3, H, W)
    std_G = data_std_G.view(1, 3, H, W)

    # standardize input data to the model
    preds = []
    input = x_seq_stdized[:, 0].clone()  # [B, C_in, H, W]

    for i in range(T-1):
        input = input * std_A + mean_A  # unnormalize
        input = (input - mean_G[:, :C_in, :, :]) / std_G[:, :C_in, :, :]  # standardize to model G
        y_t_stdized = model_G(input)  # shape: [B, C_out, H, W]

        if wet is not None:
            y_t_stdized *= wet

        preds.append(y_t_stdized.clone())

        input = input * std_G[:, :C_in, :, :] + mean_G[:, :C_in, :, :]  # unnormalize
        input = (input - mean_A) / std_A  # standardize to model A

        if pred_residual_A:
            input = input + model_A(input)  # shape: [B, C_in, H, W]
        else:
            input = model_A(input)  # shape: [B, C_in, H, W]    

    y_seq = torch.stack(preds, dim=1)  # shape: [B, T-1, C_out, H, W]
    y_seq = y_seq * std_G[None, :, C_in:, :, :] + mean_G[None, :, C_in:, :, :]  # unnormalize
    
    if added_batch_dim:
        y_seq = y_seq.squeeze(0)  # return to [T, C_out, H, W]
        out_seq_true = out_seq_true.squeeze(0)  # return to [T-1, C_in, H, W]

    return out_seq_true.cpu(), y_seq.cpu()