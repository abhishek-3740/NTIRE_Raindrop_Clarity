import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (a differentiable variant of L1 loss)

    Widely used in image restoration tasks because:
    - Robust to outliers
    - Better stability than L2
    - Directly aligned with PSNR optimization
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))