import torch
import math

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute PSNR between two tensors.

    Args:
        pred:   (B, C, H, W), range [0, 1]
        target: (B, C, H, W), range [0, 1]
        max_val: maximum pixel value (1.0 for normalized images)

    Returns:
        PSNR value in dB (float)
    """
    mse = torch.mean((pred - target) ** 2)
    if mse.item() == 0:
        return float("inf")

    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()